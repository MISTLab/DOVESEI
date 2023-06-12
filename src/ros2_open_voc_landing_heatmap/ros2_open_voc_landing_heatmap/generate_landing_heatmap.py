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


IMG_SIZE = 352
MIN_TOTAL_PIXELS = 100

class GenerateLandingHeatmap(Node):

    def __init__(self):
        super().__init__('generate_landing_heatmap')
        self.declare_parameter('prompts', "building;tree;road;water;transmission lines;post;vehicle;people")
        self.declare_parameter('prompts2dilate', "people")
        self.declare_parameter('dilation_dist', 30.0)
        self.declare_parameter('dilation_threshold', 0.5)
        self.declare_parameter('safety_threshold', 0.8)
        self.declare_parameter('model_calib_cte', 40)
        #self.add_on_set_parameters_callback(self.parameters_callback)

        self.prompts = self.get_parameter('prompts').value.split(';')
        self.prompts2dilate = self.get_parameter('prompts2dilate').value.split(';')
        self.dilation_dist = self.get_parameter('dilation_dist').value
        self.dilation_threshold = self.get_parameter('dilation_threshold').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.model_calib_cte = self.get_parameter('model_calib_cte').value

        # Generate the distance matrix (based on the output size of CLIPSeg - IMG_SIZE)
        # It has the distance from the centre of the matrix
        centre_y = IMG_SIZE/2
        centre_x = IMG_SIZE/2
        max_dist = np.sqrt((centre_y)**2+(centre_x)**2)
        local_indices = np.transpose(np.nonzero(np.ones((IMG_SIZE,IMG_SIZE), dtype=float)))
        dists = (max_dist-np.sqrt(((local_indices-[centre_y,centre_x])**2).sum(axis=1)))
        dists -= dists.min()
        dists /= dists.max()
        self.final_dists = np.zeros((IMG_SIZE,IMG_SIZE), dtype=float)
        for i,(j,k) in enumerate(local_indices):
            self.final_dists[j,k] = dists[i]

        self.cv_bridge = CvBridge()
        self.srv = self.create_service(GetLandingHeatmap, 'generate_landing_heatmap', self.get_landing_heatmap_callback)
        self.get_logger().info('generate_landing_heatmap is up and running!')

    def get_landing_heatmap_callback(self, request, response):
        # Inputs: request.image, request.px_per_m
        # Outputs: response.heatmap, response.success

        dilate_pixels = int(request.px_per_m*self.dilation_dist)

        input_image = self.cv_bridge.imgmsg_to_cv2(request.image, desired_encoding='rgb8')

        with torch.inference_mode():
            inputs = processor(text=self.prompts, images=[input_image] * len(self.prompts), padding=True, return_tensors="pt")
            for k in inputs:
                if torch.cuda.is_available():
                    inputs[k] = inputs[k].cuda()
                else:
                    inputs[k] = inputs[k]
            logits = model(**inputs).logits
            logits = logits.softmax(dim=1).detach().cpu().numpy()
        
        # Apply calibration and clip to 0,1
        logits = np.clip(logits.reshape((len(self.prompts),*logits.shape[-2:]))*self.model_calib_cte, 0, 1)

        # Dilate the class that we need a minimum distance
        # but only values above DILATE_THRS
        for pi,p in enumerate(self.prompts):
            if p in self.prompts2dilate:
                dilated = cv2.filter2D((logits[pi]>self.dilation_threshold).astype('float32'),-1,
                                        np.ones((dilate_pixels,dilate_pixels))/dilate_pixels**2)
                # Fuse the dilated with the original using max values
                # (the 0.01 should be 0.0, but it's needed to avoid the noise)
                logits[pi] = np.vstack(([logits[pi]],
                                        [(dilated>0.01).astype('float32')])).max(axis=0)

        # Fuse all logits using max values
        logits = 1-logits.max(axis=0)

        # # Blur to smooth the ViT patches
        # logits = cv2.filter2D(logits,-1,np.ones((6,6))/36)*self.final_dists

        logits = (logits>self.safety_threshold).astype('float32')*self.final_dists

        # (11,11) so the center is at (5,5)
        decimated = cv2.resize(logits,(11,11),cv2.INTER_AREA)*255

        # TODO: implement some logit to decide this...
        response.success = True
        
        # returns the decimated heatmap as grayscale image
        response.heatmap = self.cv_bridge.cv2_to_imgmsg((decimated).astype('uint8'), encoding='mono8')

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