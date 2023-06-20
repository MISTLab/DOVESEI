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
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg
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
        self.declare_parameter('blur_kernel_size', 15)
        #self.add_on_set_parameters_callback(self.parameters_callback)
        img_topic = self.get_parameter('img_topic').value
        self.blur_kernel_size = self.get_parameter('blur_kernel_size').value

        # Generate the distance matrix (based on the output size of CLIPSeg - CLIPSEG_OUTPUT_SIZE)
        # It has the distance from the centre of the matrix
        centre_y = CLIPSEG_OUTPUT_SIZE/2
        centre_x = CLIPSEG_OUTPUT_SIZE/2
        max_dist = np.sqrt((centre_y)**2+(centre_x)**2)
        local_indices = np.transpose(np.nonzero(np.ones((CLIPSEG_OUTPUT_SIZE,CLIPSEG_OUTPUT_SIZE), dtype=float)))
        dists = (max_dist-np.sqrt(((local_indices-[centre_y,centre_x])**2).sum(axis=1)))
        dists -= dists.min()
        dists /= dists.max()
        self.final_dists = np.zeros((CLIPSEG_OUTPUT_SIZE,CLIPSEG_OUTPUT_SIZE), dtype=float)
        for i,(j,k) in enumerate(local_indices):
            self.final_dists[j,k] = dists[i]

        self.cv_bridge = CvBridge()
        self.last_img = None

        qos_prof = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1)
        self.img_sub = self.create_subscription(
            ImageMsg,
            img_topic,
            self.img_sub_cb, qos_profile=qos_prof)
        
        self.srv = self.create_service(GetLandingHeatmap, 'generate_landing_heatmap', self.get_landing_heatmap_callback)
        if torch.cuda.is_available(): self.get_logger().warn('generate_landing_heatmap is using cuda!')
        self.get_logger().info('generate_landing_heatmap is up and running!')


    def img_sub_cb(self, msg):
            self.last_img = cv2.resize(self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'),(CLIPSEG_OUTPUT_SIZE,CLIPSEG_OUTPUT_SIZE))

    @staticmethod
    def get_mask(img, label):
        # img is BGR!
        return (img[...,2] == LABELS[label][0]) & (img[...,1] == LABELS[label][1]) & (img[...,0] == LABELS[label][2])
    
    def get_landing_heatmap_callback(self, request, response):
        # Inputs: request.image, request.prompts, request.erosion_size
        # Outputs: response.heatmap, response.success

        if self.last_img is None:
            response.success = False
            self.get_logger().warn(f'Empty response!')
            return response

        input_image = self.cv_bridge.imgmsg_to_cv2(request.image, desired_encoding='rgb8')
        prompts = request.prompts.split(';')
        erosion_size = request.erosion_size

        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                            (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))
       
        # places to land mask
        logits = np.ones(self.last_img.shape[:2])==0
        for label in PLACES2LAND:
            logits |= self.get_mask(self.last_img, label)
        logits = logits.astype('float32')

        # logits = self.get_mask(self.last_img, "SideWalk").astype('float32')

        # Blur to smooth the ViT patches
        logits = cv2.blur(logits,(self.blur_kernel_size,self.blur_kernel_size))

    
        logits = cv2.erode(logits, element)

        # Apply the distance gradient
        logits = logits*self.final_dists

        # Finally, resize to match input image (CLIPSeg resizes without keeping the proportions)
        logits = cv2.resize(logits, (input_image.shape[1],input_image.shape[0]), cv2.INTER_AREA)

        # and convert to a grayscale image (0 to 255)
        logits = (logits*255).astype('uint8')

        # TODO: implement some logit to decide this...
        response.success = True
        
        # returns heatmap as grayscale image
        response.heatmap = self.cv_bridge.cv2_to_imgmsg(logits, encoding='mono8')

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