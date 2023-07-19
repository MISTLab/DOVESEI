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
        self.declare_parameter('model_calib_cte', 2.0)
        self.declare_parameter('blur_kernel_size', 15)
        self.declare_parameter('prompt_engineering', "{}")
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.model_calib_cte = self.get_parameter('model_calib_cte').value
        self.blur_kernel_size = self.get_parameter('blur_kernel_size').value
        self.prompt_engineering = self.get_parameter('prompt_engineering').value

        self.cv_bridge = CvBridge()

        self.final_img_pub = self.create_publisher(ImageMsg, "/final_heatmap",1) ##DEBUG

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
        negative_prompts = [w for w in request.negative_prompts.split(';') if len(w)]
        positive_prompts = [w for w in request.positive_prompts.split(';') if len(w)]
        assert len(negative_prompts)>0, "You must supply at least one negative prompt!"
        assert len(positive_prompts)>0, "You must supply at least one positive prompt!"
        prompts = negative_prompts + positive_prompts
        prompts = [self.prompt_engineering.format(p) for p in prompts]

        with torch.inference_mode():
            # TODO: recycle old prompts to avoid encoding the same prompts multiple times...
            inputs = processor(text=prompts, images=[input_image] * len(prompts), padding=True, return_tensors="pt")
            for k in inputs:
                if torch.cuda.is_available():
                    inputs[k] = inputs[k].cuda()
                else:
                    inputs[k] = inputs[k]
            logits = model(**inputs).logits
            logits = logits.detach().cpu().numpy()

        # Filters prompts according to their max values
        logits = logits.argmax(axis=0)

        # Keep only the positive prompts
        logits = np.logical_or.reduce([logits == n for n in np.arange(len(prompts))[-len(positive_prompts):]])

        logits = logits.astype('uint8')*255

        # Blur to smooth the ViT patches
        logits = cv2.blur(logits,(self.blur_kernel_size,self.blur_kernel_size))
 
        # Finally, resize to match input image (CLIPSeg resizes without keeping the proportions)
        logits = cv2.resize(logits, (input_image.shape[1],input_image.shape[0]), cv2.INTER_AREA)

        # TODO: implement some logic to decide this...
        response.success = True
        
        # returns heatmap as grayscale image
        response.heatmap = self.cv_bridge.cv2_to_imgmsg(logits, encoding='mono8')
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