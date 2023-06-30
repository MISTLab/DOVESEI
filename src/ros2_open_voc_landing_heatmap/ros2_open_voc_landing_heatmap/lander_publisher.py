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


FOV = math.radians(73) #TODO: get this from the camera topic...

PROMPTS_SEARCHING = ["building", "tree", "road", "water", "transmission lines", "lamp post", "vehicle", "people"]
PROMPTS_DESCENDING = ["vehicle", "people"]
EPS = 0.001
MAX_SEG_HEIGHT = 17 # helps filtering noise
XY_GIVEUP_SEARCH_TIME = 60
RANDOM_GIVEUP_SEARCH = False
class TwistPublisher(Node):

    def __init__(self):
        super().__init__('lander_publisher')
        self.declare_parameter('img_topic', '/carla/flying_sensor/rgb_down/image')
        self.declare_parameter('depth_topic', '/carla/flying_sensor/depth_down/image')
        self.declare_parameter('heatmap_topic', '/heatmap')
        self.declare_parameter('depth_proj_topic', '/depth_proj')
        self.declare_parameter('twist_topic', '/quadctrl/flying_sensor/ctrl_twist_sp')
        self.declare_parameter('mov_avg_size', 20)
        self.declare_parameter('gain', 20)
        self.declare_parameter('z_speed', 1.0)
        self.declare_parameter('depth_new_size', 100)
        self.declare_parameter('depth_smoothness', 0.2)
        self.declare_parameter('mean_depth_side', 20)
        self.declare_parameter('altitude_landed', 1)
        self.declare_parameter('safe_altitude', 50)
        self.declare_parameter('safety_radius', 2.0)
        self.declare_parameter('giveup_after_sec', 5)
        self.declare_parameter('max_depth_sensing', 20)
        img_topic = self.get_parameter('img_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        self.heatmap_topic = self.get_parameter('heatmap_topic').value
        self.depth_proj_topic = self.get_parameter('depth_proj_topic').value
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
        self.giveup_after_sec = self.get_parameter('giveup_after_sec').value
        self.max_depth_sensing = self.get_parameter('max_depth_sensing').value
        

        
        self.mov_avg_counter = 0

        self.heatmap_mov_avg = None

        self.img_msg = None

        self.landing_done = False

        self.giveup_timer = 0

        self.giveup_search_timer = 0

        self.giveup_xy_search = (0,0)

        self.cli = self.create_client(GetLandingHeatmap, 'generate_landing_heatmap',
                                      callback_group=ReentrantCallbackGroup())
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('generate_landing_heatmap service not available, waiting again...')
        self.req = GetLandingHeatmap.Request()
        self.cv_bridge = CvBridge()

        # # QoS profile that will only keep the last message
        # # qos_prof = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        # qos_prof = QoSProfile(history=QoSHistoryPolicy.KEEP_LAST, depth=1)
                
        self.twist_pub = self.create_publisher(Twist, self.twist_topic,1)
        self.heatmap_pub = self.create_publisher(ImageMsg, self.heatmap_topic,1)
        self.depth_proj_pub = self.create_publisher(ImageMsg, self.depth_proj_topic,1)

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

        safety_radius_pixels = int(self.safety_radius/(2*proj/depth.shape[1]))
        mask = np.zeros_like(depth)
        mask = cv2.circle(mask, (depth_center[1],depth_center[0]), safety_radius_pixels, 255, -1)
        depth[mask!=255] = 0
        
        img_msg = self.cv_bridge.cv2_to_imgmsg(depth.astype('uint8'), encoding='mono8')
        self.depth_proj_pub.publish(img_msg)
        return depth[depth>0].std(),depth[depth>0].min()



    def error_from_semantics(self, rgbmsg, altitude, prompts):
        proj = math.tan(FOV/2)*altitude
        self.get_logger().debug(f'Sending heatmap service request...')
        response = self.get_heatmap(rgbmsg, 
                                    prompts,
                                    7)
        if response is None:
            self.get_logger().error(f'Empty response?!?!')
            return
        
        self.get_logger().debug(f'Heatmap received!')
        heatmap_msg = response.heatmap
        heatmap = self.cv_bridge.imgmsg_to_cv2(heatmap_msg, desired_encoding='mono8')
        
        # Debug
        # heatmap = np.zeros(heatmap.shape, dtype=heatmap.dtype)
        # heatmap[:50,:50] = 255
        # heatmap[-50:,-50:] = 255

        safety_radius_pixels = int(self.safety_radius/(2*proj/heatmap.shape[1]))
        resize = heatmap.shape[0]//safety_radius_pixels
        resize += 1-(resize % 2) # always odd
        resize = resize if resize < MAX_SEG_HEIGHT else MAX_SEG_HEIGHT
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
        xy_idx = np.dstack(np.unravel_index(np.argsort(heatmap_resized.ravel()), heatmap_resized.shape))[0][::-1].astype('float16')
        xy_idx[:,0] =  (-(xy_idx[:,0] - int(heatmap_center[0]))) / heatmap_center[0]
        xy_idx[:,1] = (xy_idx[:,1] - int(heatmap_center[1])) / heatmap_center[1]

        self.get_logger().debug(f'Publishing resized heatmap image at {self.heatmap_topic}')
        img_msg = self.cv_bridge.cv2_to_imgmsg(heatmap_resized, encoding='mono8')
        self.heatmap_pub.publish(img_msg)
        return xy_idx


    def get_rangefinder(self):
        res = self.get_tf()
        if res is None:
            return None
        t, init_pos, init_quat = res
        return init_pos[2]
    

    def shutdown_after_landed(self):
        # TODO: implement procedures after landed
        self.get_logger().error("Landed!")


    def sense_and_act(self, rgbmsg, depthmsg):
        # Beware: spaghetti-code state machine below!
        self.get_logger().debug(f'New data received!')

        altitude = self.get_rangefinder()
        if altitude is None:
            return
        
        self.get_logger().info(f"Altitude: {altitude:.2f}")

        if self.giveup_timer == 0:
            time_since_giveup = 0
        else:
            time_since_giveup = (self.get_clock().now().nanoseconds/1E9 - self.giveup_timer)
            self.get_logger().warn(f"Time since giveup_timer enabled: {time_since_giveup}")

        x = y = z = 0.0
        if not self.landing_done:
            depth_std, depth_min = self.depth_proj(depthmsg, altitude, max_dist=self.max_depth_sensing)

            xs_err = ys_err = 0.0
            prompts = PROMPTS_SEARCHING
            if altitude < self.safe_altitude*0.8: # 0.8 is to give a margin for the heatmap generation
                self.get_logger().warn(f"Safe altitude breached, no movements allowed on XY!")    
                prompts = PROMPTS_DESCENDING

            xy_err = self.error_from_semantics(rgbmsg, altitude, prompts)
            xs_err = xy_err[0,0]
            ys_err = xy_err[0,1]
            self.get_logger().info(f"Current prompts: {prompts}")
            self.get_logger().info(f"Segmentation X,Y err: {xs_err:.2f},{ys_err:.2f}, Depth std, min: {depth_std:.2f},{depth_min:.2f}")
            
            x = xs_err
            y = ys_err

            # Very rudimentary filter
            x = x if abs(x) > EPS else 0.0
            y = y if abs(y) > EPS else 0.0

            zero_xy_error = (abs(x)+abs(y)) == 0.0
            flat_surface_below = depth_std < self.depth_smoothness
            no_collisions_ahead = (depth_min == self.max_depth_sensing) or (depth_min - altitude) < self.depth_smoothness
            give_up_landing_here = (time_since_giveup > self.giveup_after_sec)
            self.get_logger().info(f"zero_xy_error: {zero_xy_error}, flat_surface_below: {flat_surface_below}, no_collisions_ahead: {no_collisions_ahead}, give_up_landing_here: {give_up_landing_here}")

            if altitude >= self.safe_altitude:
                if give_up_landing_here:
                    if self.giveup_search_timer==0:
                        if RANDOM_GIVEUP_SEARCH:
                            self.giveup_xy_search = np.random.rand(2)-1 # uniform random [-0.5,0.5] search direction
                        else:
                            # xy_err are already ordered according to the objective function 
                            # ("emptiness" and distance to the UAV) and their values are normalized in relation to the heatmap.
                            # Then it filters values at least distant 0.5 from the UAV and gets the first as its search direction.
                            self.giveup_xy_search = xy_err[(xy_err**2).sum(axis=1) >= 0.5][0] 
                        self.giveup_search_timer = self.get_clock().now().nanoseconds/1E9
                    random_search_time_passed = (self.get_clock().now().nanoseconds/1E9 - self.giveup_search_timer)
                    if random_search_time_passed > XY_GIVEUP_SEARCH_TIME:
                        self.giveup_timer = 0 # safe place, reset giveup_timer
                        self.giveup_search_timer = 0
                        self.get_logger().error(f"Random search finished!")
                    else:
                        self.get_logger().error(f"Random search is ON ({random_search_time_passed})!")
                    x,y = self.giveup_xy_search
                elif zero_xy_error:
                    z = -self.z_speed  
            else:
                if zero_xy_error and flat_surface_below and no_collisions_ahead and not give_up_landing_here:
                    self.giveup_timer = 0 # things look fine, reset give up timer
                    z = -self.z_speed
                else:
                    if self.giveup_timer == 0:
                        self.giveup_timer = self.get_clock().now().nanoseconds/1E9
                    z = self.z_speed
                x = y = 0.0 # below safe altitude, no movements allowed on XY


        if altitude<self.altitude_landed:
            self.landing_done = True
            self.shutdown_after_landed()

        twist = Twist()
        # The UAV's maximum bank angle is limited to a very small value
        # and this is why such a simple control works.
        # Additionally, the assumption is that the maximum speed is very low
        # otherwise the moving average used in the semantic segmentation will break.
        twist.linear.x = x * self.gain
        twist.linear.y = -y * self.gain
        twist.linear.z = z
        # twist.linear.x = twist.linear.y = twist.linear.z = 0.0 ##DEBUG
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        self.get_logger().info(f'Publishing velocities ({(twist.linear.x, twist.linear.y, twist.linear.z)}) at {self.twist_topic}')
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
        self.get_logger().error('Shutting down... sending zero velocities!')
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