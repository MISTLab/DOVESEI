import numpy as np

import cv2

from sensor_msgs.msg import Image as ImageMsg
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        
        self.declare_parameter('image_path', 'carla.png')
        self.declare_parameter('topic', '/carla/flying_sensor/rgb_down/image')
        self.declare_parameter('delta_t', 1.0)
        self.declare_parameter('output_size', 352)
        self.image_path = self.get_parameter('image_path').value
        self.topic = self.get_parameter('topic').value
        delta_t = self.get_parameter('delta_t').value
        output_size = self.get_parameter('output_size').value
        

        self.cv_bridge = CvBridge()
        img_np = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)
        img_np = cv2.resize(img_np,(output_size,output_size))

        self.img_msg = self.cv_bridge.cv2_to_imgmsg(img_np, encoding='rgb8')
        
        self.img_pub = self.create_publisher(ImageMsg, self.topic,1)

        self.img_timer = self.create_timer(delta_t, self.on_img_timer)

    def on_img_timer(self):
        self.img_pub.publish(self.img_msg)


def main():
    rclpy.init()
    image_publisher = ImagePublisher()
    image_publisher.get_logger().info(f'Publishing image {image_publisher.image_path} at topic {image_publisher.topic}')
    try:
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.get_logger().info(f'Shutting image_publisher down...')
        rclpy.shutdown()


if __name__ == '__main__':
    main()