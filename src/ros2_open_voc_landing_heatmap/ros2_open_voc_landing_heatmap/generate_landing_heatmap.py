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
from cv_bridge import CvBridge


CLIPSEG_OUTPUT_SIZE = 352

class GenerateLandingHeatmap(Node):

    def __init__(self):
        super().__init__('generate_landing_heatmap')
        self.declare_parameter('model_calib_cte', 0.0)
        self.declare_parameter('blur_kernel_size', 15)
        self.declare_parameter('prompt_engineering', "3D rendered image of {}, animation, game")
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.model_calib_cte = self.get_parameter('model_calib_cte').value
        self.blur_kernel_size = self.get_parameter('blur_kernel_size').value
        self.prompt_engineering = self.get_parameter('prompt_engineering').value

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

        self.positive_img_pub = self.create_publisher(ImageMsg, "/original_heatmap_positive",1) ##DEBUG
        self.negative_img_pub = self.create_publisher(ImageMsg, "/original_heatmap_negative",1) ##DEBUG

        self.srv = self.create_service(GetLandingHeatmap, 'generate_landing_heatmap', self.get_landing_heatmap_callback)
        if torch.cuda.is_available(): self.get_logger().warn('generate_landing_heatmap is using cuda!')
        self.get_logger().info('generate_landing_heatmap is up and running!')


    def parameters_callback(self, params):
        for param in params:
            try:
                var_type = type(getattr(self, param.name))
                setattr(self, param.name, var_type(param.value))
                self.get_logger().info(f'Parameter updated: {param.name} = {param.value}')
            except AttributeError:
                return SetParametersResult(successful=False)
        return SetParametersResult(successful=True)
        
    

    def get_landing_heatmap_callback(self, request, response):
        # Inputs: request.image, request.prompts, request.erosion_size
        # Outputs: response.heatmap, response.success

        self.get_logger().debug('New heatmap request received!')

        input_image_msg = request.image
        input_image = self.cv_bridge.imgmsg_to_cv2(input_image_msg, desired_encoding='rgb8')
        negative_prompts = request.negative_prompts.split(';')
        positive_prompts = request.positive_prompts.split(';')
        prompts = negative_prompts + positive_prompts
        prompts = [self.prompt_engineering.format(p) for p in prompts]
        erosion_size = request.erosion_size
        safety_threshold = request.safety_threshold

        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                            (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))

        with torch.inference_mode():
            # TODO: recycle old prompts to avoid encoding the same prompts multiple times...
            inputs = processor(text=prompts, images=[input_image] * len(prompts), padding=True, return_tensors="pt")
            for k in inputs:
                if torch.cuda.is_available():
                    inputs[k] = inputs[k].cuda()
                else:
                    inputs[k] = inputs[k]
            logits = model(**inputs).logits
            logits = logits.softmax(dim=1).detach().cpu().numpy()

        # Apply calibration (offset) and clip to [0,1]
        logits = np.clip(logits.reshape((len(prompts),*logits.shape[-2:]))+self.model_calib_cte, 0, 1)

        # Fuse all logits using the max values
        if len(negative_prompts):
            # Normalise individual prompts
            logits[:len(negative_prompts)] = logits[:len(negative_prompts)]/logits[:len(negative_prompts)].max(axis=1).max(axis=1)[:,None,None]
            # Fuse using their max values
            fused_negative_logits = logits[:len(negative_prompts)].max(axis=0)
        else:
            fused_negative_logits = np.zeros(logits.shape[-2:])

        if len(positive_prompts):
            # Normalise individual prompts
            logits[-len(positive_prompts):] = logits[-len(positive_prompts):]/logits[-len(positive_prompts):].max(axis=1).max(axis=1)[:,None,None]
            # Fuse using their max values
            fused_positive_logits = logits[-len(positive_prompts):].max(axis=0)
        else:
            fused_positive_logits = np.zeros(logits.shape[-2:])

        # Final logits will have higher values for the good places to land and lower for the bad ones
        logits = 1 + np.clip(fused_positive_logits-fused_negative_logits, -1, 0)

        positive_heatmap = (cv2.resize(fused_positive_logits, (input_image.shape[1],input_image.shape[0]), cv2.INTER_AREA)*255).astype('uint8')##DEBUG
        positive_heatmap_msg = self.cv_bridge.cv2_to_imgmsg(positive_heatmap*255, encoding='mono8')
        positive_heatmap_msg.header.frame_id = input_image_msg.header.frame_id
        self.positive_img_pub.publish(positive_heatmap_msg)##DEBUG
        negative_heatmap = (cv2.resize(fused_negative_logits, (input_image.shape[1],input_image.shape[0]), cv2.INTER_AREA)*255).astype('uint8')##DEBUG
        negative_heatmap_msg = self.cv_bridge.cv2_to_imgmsg(negative_heatmap*255, encoding='mono8')
        negative_heatmap_msg.header.frame_id = input_image_msg.header.frame_id
        self.negative_img_pub.publish(negative_heatmap_msg)##DEBUG

        # Blur to smooth the ViT patches
        logits = cv2.blur(logits,(self.blur_kernel_size,self.blur_kernel_size))

        # Creates a mask of the best places to land
        logits = (logits>safety_threshold).astype('float32')
    
        logits = cv2.erode(logits, element)

        # Apply the distance gradient
        logits = logits*self.final_dists

        # Finally, resize to match input image (CLIPSeg resizes without keeping the proportions)
        logits = cv2.resize(logits, (input_image.shape[1],input_image.shape[0]), cv2.INTER_AREA)

        # and convert to a grayscale image (0 to 255)
        logits = (logits*255).astype('uint8')

        # TODO: implement some logic to decide this...
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