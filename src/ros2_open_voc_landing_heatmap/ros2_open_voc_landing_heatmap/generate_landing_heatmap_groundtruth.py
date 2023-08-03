from time import sleep
import numpy as np
import cv2
from PIL import Image

import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined") 
if torch.cuda.is_available():
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").cuda()
else:
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


from ros2_open_voc_landing_heatmap_srv.srv import GetLandingHeatmap
import rclpy
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge

LABELS = {
'Unlabeled':    (0, 0, 0),
'Building':     (70, 70, 70),
'Fence':        (100, 40, 40),
'Other':        (55, 90, 80),
'Pedestrian':   (220, 20, 60),
'Pole':         (153, 153, 153),
'RoadLine':     (157, 234, 50),
'Road':         (128, 64, 128),
'SideWalk':     (244, 35, 232),
'Vegetation':   (107, 142, 35),
'Vehicles':     (0, 0, 142),
'Wall':         (102, 102, 156),
'TrafficSign':  (220, 220, 0),
'Sky':          (70, 130, 180),
'Ground':       (81, 0, 81),
'Bridge':       (150, 100, 100),
'RailTrack':    (230, 150, 140),
'GuardRail':    (180, 165, 180),
'TrafficLight': (250, 170, 30),
'Static':       (110, 190, 160),
'Dynamic':      (170, 120, 50),
'Water':        (45, 60, 150),
'Terrain':      (145, 170, 100)
}

PLACES2LAND = ["Terrain", "Ground", "Other", "SideWalk", "Unlabeled"]

CLIPSEG_OUTPUT_SIZE = 352

class GenerateLandingHeatmap(Node):
    def __init__(self):
        super().__init__('generate_landing_heatmap')
        self.declare_parameter('img_topic', '/carla/flying_sensor/semantic_segmentation_down/image')
        self.img_topic = self.get_parameter('img_topic').value
        self.cv_bridge = CvBridge()

        self.get_logger().warn('Waiting for the simulator...')
        self.check_flying_sensor_alive = self.create_subscription(
            Imu,
            '/quadsim/flying_sensor/imu',
            self.start_node,
            1)


    def start_node(self, msg):
        self.get_logger().warn('Simulator is online!')
        self.destroy_subscription(self.check_flying_sensor_alive) # we don't need this subscriber anymore...

        self.last_img = None
        self.final_img_pub = self.create_publisher(ImageMsg, "/final_heatmap",1) ##DEBUG

        self.img_sub = self.create_subscription(
            ImageMsg,
            self.img_topic,
            self.img_sub_cb, 1)

        self.srv = self.create_service(GetLandingHeatmap, 'generate_landing_heatmap', self.get_landing_heatmap_callback)
        self.get_logger().info('generate_landing_heatmap is up and running!')
        

    def img_sub_cb(self, msg):
            self.last_img = cv2.resize(self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'),(CLIPSEG_OUTPUT_SIZE,CLIPSEG_OUTPUT_SIZE))

    @staticmethod
    def get_mask(img, label):
        # img is BGR!
        return (img[...,2] == LABELS[label][0]) & (img[...,1] == LABELS[label][1]) & (img[...,0] == LABELS[label][2])
    
    def get_landing_heatmap_callback(self, request, response):
        # Inputs: request.image, request.negative_prompts, request.positive_prompts, request.prompt_engineering
        #         request.safety_threshold, request.blur_kernel_size, request.dynamic_threshold
        # Outputs: response.heatmap, response.success

        self.get_logger().debug('New heatmap request received!')

        if self.last_img is None:
            response.success = False
            self.get_logger().warn(f'Empty response!')
            return response

        input_image_msg = request.image
        input_image = self.cv_bridge.imgmsg_to_cv2(input_image_msg, desired_encoding='rgb8')
        safety_threshold = request.safety_threshold
        blur_kernel_size = request.blur_kernel_size
        
       
        # places to land mask
        logits = np.ones(self.last_img.shape[:2])==0
        for label in PLACES2LAND:
            logits |= self.get_mask(self.last_img, label)
        logits = logits.astype('float32')

        kernel = np.ones((3,3),np.uint8)
        logits = cv2.erode(logits,kernel,iterations=2)

        # Blur
        logits = cv2.blur(logits,(blur_kernel_size, blur_kernel_size))

        # Finally, resize to match input image
        logits = cv2.resize(logits, (input_image.shape[1],input_image.shape[0]), cv2.INTER_AREA)

        logits = logits > 0.2
        response.success = False
        if request.dynamic_threshold > 0.0:
            total_pixels = np.prod(logits.shape)
            if not ((logits==True).sum()/total_pixels < request.dynamic_threshold):
                response.success = True
        else:
            response.success = True
        
        # returns heatmap as grayscale image
        response.heatmap = self.cv_bridge.cv2_to_imgmsg((logits*255).astype('uint8'), encoding='mono8')
        response.heatmap.header.frame_id = input_image_msg.header.frame_id

        self.final_img_pub.publish(response.heatmap)##DEBUG

        return response


def main():
    rclpy.init()

    generate_landing_heatmap = GenerateLandingHeatmap()

    try:
        rclpy.spin(generate_landing_heatmap)
    except KeyboardInterrupt:
        pass
    finally:
        generate_landing_heatmap.get_logger().info(f'Shutting generate_landing_heatmap down...')
        rclpy.shutdown()


if __name__ == '__main__':
    main()