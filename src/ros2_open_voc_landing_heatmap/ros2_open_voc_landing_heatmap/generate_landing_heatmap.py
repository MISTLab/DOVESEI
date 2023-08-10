from time import sleep
import numpy as np
import cv2
# from PIL import Image
from sensor_msgs.msg import Image as ImageMsg ##DEBUG

import torch
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined") 
if torch.cuda.is_available():
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").cuda()
else:
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


from ros2_open_voc_landing_heatmap_srv.srv import GetLandingHeatmap
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge


CLIPSEG_OUTPUT_SIZE = 352
DYNAMIC_THRESHOLD_MAXSTEPS = 100
class GenerateLandingHeatmap(Node):

    def __init__(self):
        super().__init__('generate_landing_heatmap')
        self.cv_bridge = CvBridge()

        self.get_logger().warn('Waiting for the simulator...')
        self.check_flying_sensor_alive = self.create_subscription(
            ImageMsg,
            '/carla/flying_sensor/rgb_down/image',
            self.start_node,
            1)


    def start_node(self, msg):
        self.get_logger().warn('Simulator is online!')
        self.destroy_subscription(self.check_flying_sensor_alive) # we don't need this subscriber anymore...

        self.final_img_pub = self.create_publisher(ImageMsg, "/final_heatmap",1) ##DEBUG

        self.srv = self.create_service(GetLandingHeatmap, 'generate_landing_heatmap', self.get_landing_heatmap_callback)
        if torch.cuda.is_available(): 
            self.get_logger().warn('generate_landing_heatmap is using CUDA GPU!')
        else:
            self.get_logger().error('generate_landing_heatmap is using CPU!')
        self.get_logger().info('generate_landing_heatmap is up and running!')
        
    

    def get_landing_heatmap_callback(self, request, response):
        # Inputs: request.image, request.negative_prompts, request.positive_prompts, request.prompt_engineering
        #         request.safety_threshold, request.blur_kernel_size, request.dynamic_threshold
        # Outputs: response.heatmap, response.success

        self.get_logger().debug('New heatmap request received!')

        input_image_msg = request.image
        input_image = self.cv_bridge.imgmsg_to_cv2(input_image_msg, desired_encoding='rgb8')
        negative_prompts = [w for w in request.negative_prompts.split(';') if len(w)]
        positive_prompts = [w for w in request.positive_prompts.split(';') if len(w)]
        prompt_engineering = request.prompt_engineering
        safety_threshold = request.safety_threshold
        blur_kernel_size = request.blur_kernel_size
        
        assert len(negative_prompts)>0, "You must supply at least one negative prompt!"
        assert len(positive_prompts)>0, "You must supply at least one positive prompt!"
        prompts = negative_prompts + positive_prompts
        prompts = [prompt_engineering.format(p) for p in prompts]

        with torch.inference_mode():
            # TODO: recycle old prompts to avoid encoding the same prompts multiple times...
            inputs = processor(text=prompts, images=[input_image] * len(prompts), padding=True, return_tensors="pt")
            for k in inputs:
                if torch.cuda.is_available():
                    inputs[k] = inputs[k].cuda()
                else:
                    inputs[k] = inputs[k]
            logits = model(**inputs).logits
            logits = logits.softmax(dim=0).detach().cpu().numpy()

        # Keep only the positive prompts
        logits = logits[-len(positive_prompts):].sum(axis=0)

        # Blur to smooth the ViT patches
        logits = cv2.blur(logits,(blur_kernel_size, blur_kernel_size))
 
        # Finally, resize to match input image (CLIPSeg resizes without keeping the proportions)
        logits = cv2.resize(logits, (input_image.shape[1],input_image.shape[0]), cv2.INTER_AREA)

        
        logits_threshold = logits > safety_threshold
        curr_threshold = safety_threshold
        response.success = False
        if request.dynamic_threshold > 0.0:
            total_pixels = np.prod(logits.shape)
            threshold_step = safety_threshold/DYNAMIC_THRESHOLD_MAXSTEPS
            for ti in range(0,DYNAMIC_THRESHOLD_MAXSTEPS+1):
                if (logits_threshold==True).sum()/total_pixels < request.dynamic_threshold:
                    curr_threshold = (safety_threshold-threshold_step*ti)
                    logits_threshold = logits > curr_threshold
                else:
                    response.success = True
                    break
        else:
            response.success = True
        
        response.curr_threshold = curr_threshold

        # returns heatmap as grayscale image
        response.heatmap = self.cv_bridge.cv2_to_imgmsg((logits_threshold*255).astype('uint8'), encoding='mono8')
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