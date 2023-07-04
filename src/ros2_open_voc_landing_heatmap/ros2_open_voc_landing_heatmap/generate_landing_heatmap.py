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
from rclpy.node import Node
from cv_bridge import CvBridge


CLIPSEG_OUTPUT_SIZE = 352

class GenerateLandingHeatmap(Node):

    def __init__(self):
        super().__init__('generate_landing_heatmap')
        self.declare_parameter('safety_threshold', 0.8)
        self.declare_parameter('model_calib_cte', 40.0)
        self.declare_parameter('blur_kernel_size', 15)
        #self.add_on_set_parameters_callback(self.parameters_callback)

        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.model_calib_cte = self.get_parameter('model_calib_cte').value
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

        self.srv = self.create_service(GetLandingHeatmap, 'generate_landing_heatmap', self.get_landing_heatmap_callback)
        if torch.cuda.is_available(): self.get_logger().warn('generate_landing_heatmap is using cuda!')
        self.get_logger().info('generate_landing_heatmap is up and running!')

    def get_landing_heatmap_callback(self, request, response):
        # Inputs: request.image, request.prompts, request.erosion_size
        # Outputs: response.heatmap, response.success

        self.get_logger().info('New request received!')

        input_image = self.cv_bridge.imgmsg_to_cv2(request.image, desired_encoding='rgb8')
        prompts = request.prompts.split(';')
        erosion_size = request.erosion_size

        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                            (2 * erosion_size + 1, 2 * erosion_size + 1),
                                            (erosion_size, erosion_size))

        with torch.inference_mode():
            inputs = processor(text=prompts, images=[input_image] * len(prompts), padding=True, return_tensors="pt")
            for k in inputs:
                if torch.cuda.is_available():
                    inputs[k] = inputs[k].cuda()
                else:
                    inputs[k] = inputs[k]
            logits = model(**inputs).logits
            logits = logits.softmax(dim=1).detach().cpu().numpy()
        
        # Apply calibration and clip to [0,1]
        logits = np.clip(logits.reshape((len(prompts),*logits.shape[-2:]))*self.model_calib_cte, 0, 1)

        # Fuse all logits using the max values
        logits = 1-logits.max(axis=0)

        # Blur to smooth the ViT patches
        logits = cv2.blur(logits,(self.blur_kernel_size,self.blur_kernel_size))

        # Creates a mask of the best places to land
        logits = (logits>self.safety_threshold).astype('float32')
    
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

        self.get_logger().info('Returning!')
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