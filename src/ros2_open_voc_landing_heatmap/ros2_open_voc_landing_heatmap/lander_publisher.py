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
from rclpy.time import Time, Duration

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


from message_filters import ApproximateTimeSynchronizer, Subscriber


from cv_bridge import CvBridge

GIVEUPAT = 10
FOV = math.radians(73)
EPS = 0.001
class TwistPublisher(Node):

    def __init__(self):
        super().__init__('lander_publisher')
        self.declare_parameter('img_topic', '/carla/flying_sensor/rgb_down/image')
        self.declare_parameter('depth_topic', '/carla/flying_sensor/depth_down/image')
        self.declare_parameter('heatmap_topic', '/heatmap')
        self.declare_parameter('depth_cluster_topic', '/depth_cluster')
        self.declare_parameter('twist_topic', '/quadctrl/flying_sensor/ctrl_twist_sp')
        self.declare_parameter('mov_avg_size', 10)
        self.declare_parameter('gain', 20)
        self.declare_parameter('z_speed', 1.0)
        self.declare_parameter('depth_new_size', 100)
        self.declare_parameter('depth_smoothness', 0.2)
        self.declare_parameter('mean_depth_side', 20)
        self.declare_parameter('altitude_landed', 1)
        self.declare_parameter('safe_altitude', 80)
        self.declare_parameter('safety_radius', 1.0)
        img_topic = self.get_parameter('img_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.heatmap_topic = self.get_parameter('heatmap_topic').value
        self.depth_cluster_topic = self.get_parameter('depth_cluster_topic').value
        self.twist_topic = self.get_parameter('twist_topic').value
        self.mov_avg_size = self.get_parameter('mov_avg_size').value
        self.gain = self.get_parameter('gain').value
        self.z_speed = self.get_parameter('z_speed').value
        self.depth_new_size = self.get_parameter('depth_new_size').value
        self.depth_smoothness = self.get_parameter('depth_smoothness').value
        self.mean_depth_side = self.get_parameter('mean_depth_side').value
        self.altitude_landed = self.get_parameter('altitude_landed').value
        self.safe_altitude = self.get_parameter('safe_altitude').value
        self.safety_radius = self.get_parameter('safety_radius').value
        

        
        self.mov_avg_counter = 0

        self.heatmap_mov_avg = None

        self.img_msg = None

        self.landing_done = False

        self.giveup_timer = None

        self.cli = self.create_client(GetLandingHeatmap, 'generate_landing_heatmap',
                                      callback_group=ReentrantCallbackGroup())
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('generate_landing_heatmap service not available, waiting again...')
        self.req = GetLandingHeatmap.Request()
        self.cv_bridge = CvBridge()

        # # QoS profile that will only keep the last message
        # # qos_prof = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        # qos_prof = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1)
                
        self.twist_pub = self.create_publisher(Twist, self.twist_topic,1)
        self.heatmap_pub = self.create_publisher(ImageMsg, self.heatmap_topic,1)
        self.depth_cluster_pub = self.create_publisher(ImageMsg, self.depth_cluster_topic,1)

        self.tf_trials = 5
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        queue_size = 2
        delay_btw_msgs = 0.02
        tss = ApproximateTimeSynchronizer(
            [Subscriber(self, ImageMsg, img_topic),
             Subscriber(self, ImageMsg, depth_topic)],
             queue_size,
             delay_btw_msgs
             )
        
        tss.registerCallback(self.sense_and_act)

        self.get_logger().info('Ready to publish some twist messages!')


    def get_tf(self, t=0.0, timeout=1.0, map_frame="map", target_frame="flying_sensor"):
        try:
            now = Time(nanoseconds=t)
            trans = self.tf_buffer.lookup_transform(map_frame, target_frame,
                now,
                timeout=Duration(seconds=timeout))

            self.get_logger().debug(f'TF received {trans}')
            curr_pos = [trans.transform.translation.x, 
                        trans.transform.translation.y, 
                        trans.transform.translation.z]

            curr_quat = [trans.transform.rotation.x,
                        trans.transform.rotation.y,
                        trans.transform.rotation.z,
                        trans.transform.rotation.w]

            s = trans.header.stamp.sec
            ns = trans.header.stamp.nanosec
            return (s + ns/1E9), curr_pos, curr_quat

        except TransformException as ex:
            self.get_logger().error(f'Could not transform {map_frame} to {target_frame}: {ex}')


    def depth_proj(self, depthmsg, altitude, max_dist=20):
        proj = math.tan(FOV/2)*altitude

        depth = self.cv_bridge.imgmsg_to_cv2(depthmsg, desired_encoding='passthrough')
        depth = np.asarray(cv2.resize(depth, 
                                      (int(self.depth_new_size*depth.shape[1]/depth.shape[0]),self.depth_new_size),
                                      cv2.INTER_AREA))
        # In CARLA the depth goes up to 1000m, but we want up to 20m
        depth[depth>max_dist] = np.nan
        depth[np.isnan(depth)] = max_dist

        depth_center = depth.shape[0]//2, depth.shape[1]//2

        safety_radius_pixels = int(2*self.safety_radius/(proj/depth.shape[1]))
        mask = np.zeros_like(depth)
        mask = cv2.circle(mask, (depth_center[1],depth_center[0]), safety_radius_pixels, (255,255,255), -1)
        depth = cv2.bitwise_and(depth, mask)
        
        img_msg = self.cv_bridge.cv2_to_imgmsg(depth.astype('uint8'), encoding='mono8')
        self.depth_cluster_pub.publish(img_msg)
        return depth[depth>0].std(),depth[depth>0].mean()



    def error_from_semantics(self, rgbmsg, altitude, prompts):
        proj = math.tan(FOV/2)*altitude
        self.get_logger().warn(f'Sending heatmap service request...')
        response = self.get_heatmap(rgbmsg, 
                                    prompts,
                                    7)
        if response is None:
            self.get_logger().warn(f'Empty response?!?!')
            return
        
        self.get_logger().warn(f'Heatmap received!')
        heatmap_msg = response.heatmap
        heatmap = self.cv_bridge.imgmsg_to_cv2(heatmap_msg, desired_encoding='mono8')
        
        # Debug
        # heatmap = np.zeros(heatmap.shape, dtype=heatmap.dtype)
        # heatmap[:50,:50] = 255
        # heatmap[-50:,-50:] = 255

        safety_radius_pixels = int(2*self.safety_radius/(proj/heatmap.shape[1]))
        resize = heatmap.shape[0]//safety_radius_pixels
        resize += 1-(resize % 2) # always odd
        resize_w = int(resize*(heatmap.shape[1]/heatmap.shape[0]))
        resize_w = resize_w + 1-(resize_w % 2) # always odd

        if self.heatmap_mov_avg is None:
            # like an RGB image so opencv can easily resize...
            self.heatmap_mov_avg = np.zeros((resize, resize_w, self.mov_avg_size), dtype='uint8')
        
        self.heatmap_mov_avg = cv2.resize(self.heatmap_mov_avg,
                                            (resize_w, resize),cv2.INTER_AREA)
        heatmap_resized = cv2.resize(heatmap,
                                     (self.heatmap_mov_avg.shape[1],self.heatmap_mov_avg.shape[0]),cv2.INTER_AREA)
        
        self.heatmap_mov_avg[...,self.mov_avg_counter] = heatmap_resized
        heatmap_resized = self.heatmap_mov_avg.mean(axis=2).astype('uint8')
        if self.mov_avg_counter < (self.mov_avg_size-1):
            self.mov_avg_counter += 1
        else:
            self.mov_avg_counter = 0

        heatmap_center = heatmap_resized.shape[0]/2, heatmap_resized.shape[1]/2
        # descending order, best landing candidates
        x_idx, y_idx = np.dstack(np.unravel_index(np.argsort(heatmap_resized.ravel()), heatmap_resized.shape))[0][::-1][0]

        x = (-(x_idx - int(heatmap_center[0]))) / heatmap_center[0]
        y = (y_idx - int(heatmap_center[1])) / heatmap_center[1]

        self.get_logger().warn(f'Publishing resized heatmap image at {self.heatmap_topic}')
        img_msg = self.cv_bridge.cv2_to_imgmsg(heatmap_resized, encoding='mono8')
        self.heatmap_pub.publish(img_msg)
        return x,y


    def get_rangefinder(self):
        res = self.get_tf()
        if res is None:
            return None
        t, init_pos, init_quat = res
        return init_pos[2]


    def sense_and_act(self, rgbmsg, depthmsg):
        self.get_logger().warn(f'New data received!')

        altitude = self.get_rangefinder()
        if altitude is None:
            return
        
        x = y = z = 0.0
        if not self.landing_done:
            depth_std, depth_mean = self.depth_proj(depthmsg, altitude)

            xs_err = ys_err = 0.0
            prompts = ["building", "tree", "road", "water", "transmission lines", "lamp post", "vehicle", "people"]
            if altitude < self.safe_altitude:
                self.get_logger().info(f"Safe altitude reached!")    
                prompts = ["vehicle", "people"]

            xs_err,ys_err = self.error_from_semantics(rgbmsg, altitude, prompts)
            self.get_logger().info(f"Current prompts: {prompts}")
            self.get_logger().info(f"Segmentation X,Y err: {xs_err:.2f},{ys_err:.2f}, Depth std, mean: {depth_std:.2f},{depth_mean:.2f}")
            self.get_logger().info(f"Altitude: {altitude:.2f}")
            
            x = xs_err
            y = ys_err

            x = x if abs(x) > EPS else 0.0
            y = y if abs(y) > EPS else 0.0

            if altitude >= self.safe_altitude:
                z = -self.z_speed if (abs(x)+abs(y))==0.0 else 0.0
                self.giveup_timer = 0
            else:
                if (abs(x)+abs(y))==0.0 and depth_std < self.depth_smoothness and self.giveup_timer < GIVEUPAT:
                    z = -self.z_speed
                else:
                    z = self.z_speed
                    self.giveup_timer += 1


        if altitude<self.altitude_landed:
            x = y = z = 0.0
            self.get_logger().warn("Landed!")
            self.landing_done = True

        twist = Twist()
        twist.linear.x = x * self.gain
        twist.linear.y = -y * self.gain
        twist.linear.z = z
        # twist.linear.x = twist.linear.y = twist.linear.z = 0.0 ##DEBUG
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        self.get_logger().warn(f'Publishing velocities ({(twist.linear.x, twist.linear.y, twist.linear.z)}) at {self.twist_topic}')
        self.twist_pub.publish(twist)


        

    def get_heatmap(self, image_msg, prompts, erosion_size):
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
        
        event.wait(timeout=5)

        return future.result()


    def on_shutdown_cb(self):
        self.get_logger().warn('Shutting down... sending zero velocities!')
        self.twist_pub.publish(Twist())


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
        lander_publisher.on_shutdown_cb()
        executor.shutdown()
        lander_publisher.destroy_node()


if __name__ == '__main__':
    main()