from time import sleep
from threading import Lock
import math
from threading import Event

import numpy as np
import cv2


from ros2_open_voc_landing_heatmap_srv.srv import GetLandingHeatmap
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Twist

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

from cv_bridge import CvBridge

class TwistPublisher(Node):

    def __init__(self):
        super().__init__('lander_publisher')
        self.declare_parameter('img_topic', '/carla/flying_sensor/rgb_down/image')
        self.declare_parameter('heatmap_topic', '/heatmap')
        self.declare_parameter('twist_topic', '/quadctrl/flying_sensor/ctrl_twist_sp')
        self.declare_parameter('mov_avg_size', 5)
        self.declare_parameter('resize', 15)
        self.declare_parameter('max_speed', 10)
        img_topic = self.get_parameter('img_topic').value
        self.heatmap_topic = self.get_parameter('heatmap_topic').value
        self.twist_topic = self.get_parameter('twist_topic').value
        self.mov_avg_size = self.get_parameter('mov_avg_size').value
        self.resize = self.get_parameter('resize').value
        self.speed = self.get_parameter('max_speed').value

        assert (self.resize % 2) == 1, self.get_logger().error('resize parameter MUST be odd!')

        
        self.mov_avg_counter = 0

        self.heatmap_mov_avg = None

        self.cli = self.create_client(GetLandingHeatmap, 'generate_landing_heatmap',
                                      callback_group=ReentrantCallbackGroup())
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('generate_landing_heatmap service not available, waiting again...')
        self.req = GetLandingHeatmap.Request()
        self.cv_bridge = CvBridge()

        # QoS profile that will only keep the last message
        qos_prof = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.img_sub = self.create_subscription(
            ImageMsg,
            img_topic,
            self.img_sub_cb,
            qos_profile=qos_prof)
        
        self.twist_pub = self.create_publisher(Twist, self.twist_topic,1)
        self.heatmap_pub = self.create_publisher(ImageMsg, self.heatmap_topic,1)
        

        self.img_msg = None
        self.lock = Lock()

        self.get_logger().info('ready to publish some twist messages!')


    def img_sub_cb(self, msg):

            response = self.send_request(msg, 
                                        ["building", "tree", "road", "water", "transmission lines", "lamp post", "vehicle", "people"],
                                        7)
            
            self.get_logger().warn(f'Heatmap received')
            heatmap_msg = response.heatmap
            heatmap = self.cv_bridge.imgmsg_to_cv2(heatmap_msg, desired_encoding='mono8')

            if self.heatmap_mov_avg is None:
                resize_w = int(self.resize*(heatmap.shape[1]/heatmap.shape[0]))
                resize_w = resize_w + 1-(resize_w % 2) # always odd
                self.heatmap_mov_avg = np.zeros((self.mov_avg_size, self.resize, resize_w), dtype='uint8')
            
            heatmap_resized = cv2.resize(heatmap,(self.heatmap_mov_avg.shape[2],self.heatmap_mov_avg.shape[1]),cv2.INTER_AREA)

            self.heatmap_mov_avg[self.mov_avg_counter] = heatmap_resized
            heatmap_resized = self.heatmap_mov_avg.mean(axis=0).astype('uint8')
            if self.mov_avg_counter < (self.mov_avg_size-1):
                self.mov_avg_counter += 1
            else:
                self.mov_avg_counter = 0

            heatmap_center = heatmap_resized.shape[0]/2, heatmap_resized.shape[1]/2

            # descending order, best landing candidates
            x_idx,y_idx = np.dstack(np.unravel_index(np.argsort(heatmap_resized.ravel()), heatmap_resized.shape))[0][::-1][0]

            x = -(x_idx + 1 - (int(heatmap_center[0])+1))
            y = y_idx + 1 - (int(heatmap_center[1])+1)

            twist = Twist()
            twist.linear.x = x/heatmap_center[0]* self.speed
            twist.linear.y = -y/heatmap_center[1] * self.speed
            twist.linear.z = 0.0
            twist.angular.x = 0.0
            twist.angular.y = 0.0
            twist.angular.z = 0.0

            self.get_logger().warn(f'Best spot found ({x,y}). Publishing velocities ({(twist.linear.x, twist.linear.y)}) at {self.twist_topic}')
            self.twist_pub.publish(twist)
            self.get_logger().warn(f'Publishing resized heatmap image at {self.heatmap_topic}')
            img_msg = self.cv_bridge.cv2_to_imgmsg(heatmap_resized, encoding='mono8')
            self.heatmap_pub.publish(img_msg)


        

    def send_request(self, image_msg, prompts, erosion_size):
        #request.image, request.prompts, request.erosion_size
        self.req.image = image_msg
        # the service expects a string of prompts separated by ';'
        self.req.prompts = ";".join(prompts)
        self.req.erosion_size = int(erosion_size)       

        event = Event()
        def future_done_callback(future):
            event.set()

        future = self.cli.call_async(self.req)
        future.add_done_callback(future_done_callback)
        event.wait()

        return future.result()


def main():
    rclpy.init()
    lander_publisher = TwistPublisher()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(lander_publisher)
    try:
        executor.spin()

    except KeyboardInterrupt:
        pass

    finally:
        executor.shutdown()
        lander_publisher.destroy_node()


if __name__ == '__main__':
    main()