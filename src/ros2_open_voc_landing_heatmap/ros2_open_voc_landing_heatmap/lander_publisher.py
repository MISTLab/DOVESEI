import math
from threading import Event

import numpy as np
import cv2


from ros2_open_voc_landing_heatmap_srv.srv import GetLandingHeatmap
from std_msgs.msg import String
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


# We use the same FOV from the camera in the stereo pair for the 
# semantic sementation because the second doesn't need a precise projection
FOV = math.radians(73) #TODO: get this from the camera topic...

PROMPTS_SEARCHING = ["building", "tree", "road", "water", "wall", "fence", "transmission lines", "lamp post", "vehicle", "people"]
PROMPTS_DESCENDING = ["vehicle", "people"]
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
        self.declare_parameter('altitude_landed', 1)
        self.declare_parameter('safe_altitude', 50)
        self.declare_parameter('safety_radius', 2.0)
        self.declare_parameter('safety_threshold', 0.8)
        self.declare_parameter('giveup_after_sec', 5)
        self.declare_parameter('max_depth_sensing', 20)
        self.declare_parameter('use_random_search4new_place', False)
        self.declare_parameter('heatmap_mask_erosion', 2)
        self.declare_parameter('search4new_place_max_time', 60)
        self.declare_parameter('max_seg_height', 17)
        self.declare_parameter('zero_error_eps', 0.001)
        self.declare_parameter('max_landing_time_sec', 5*60)
        self.declare_parameter('min_conservative_gain', 0.1)
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
        self.altitude_landed = self.get_parameter('altitude_landed').value
        self.safe_altitude = self.get_parameter('safe_altitude').value
        self.safety_radius = self.get_parameter('safety_radius').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.giveup_after_sec = self.get_parameter('giveup_after_sec').value
        self.max_depth_sensing = self.get_parameter('max_depth_sensing').value
        self.use_random_search4new_place = self.get_parameter('use_random_search4new_place').value
        self.heatmap_mask_erosion = self.get_parameter('heatmap_mask_erosion').value
        self.search4new_place_max_time = self.get_parameter('search4new_place_max_time').value
        self.max_seg_height = self.get_parameter('max_seg_height').value
        self.zero_error_eps = self.get_parameter('zero_error_eps').value
        self.max_landing_time_sec = self.get_parameter('max_landing_time_sec').value
        self.min_conservative_gain = self.get_parameter('min_conservative_gain').value
        

        
        self.mov_avg_counter = 0

        self.heatmap_mov_avg = None

        self.img_msg = None

        self.landing_done = False

        self.giveup_landing_timer = 0

        self.search4new_place_timer = 0

        self.search4new_place_direction = (0,0)

        self.init_time_sec = self.get_clock().now().nanoseconds/1E9
        self.prev_time_sec = self.get_clock().now().nanoseconds/1E9

        self.lander_base_state_msgs = ["SEARCHING", "RESTARTING", "LANDED"]
        self.lander_base_state = None # only one base state is allowed
        self.lander_sub_state_msgs = ["SAFE_ALTITUDE", "ZERO_XY_ERROR", "FLAT_SURFACE_BELOW", "NO_COLLISIONS_AHEAD"]
        self.lander_sub_state = [] # multiple sub states

        self.cli = self.create_client(GetLandingHeatmap, 'generate_landing_heatmap',
                                      callback_group=ReentrantCallbackGroup())
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('generate_landing_heatmap service not available, waiting again...')
        self.req = GetLandingHeatmap.Request()
        self.cv_bridge = CvBridge()
                
        self.twist_pub = self.create_publisher(Twist, self.twist_topic,1)
        self.heatmap_pub = self.create_publisher(ImageMsg, self.heatmap_topic,1)
        self.depth_proj_pub = self.create_publisher(ImageMsg, self.depth_proj_topic,1)
        self.state_pub = self.create_publisher(String, 'lander_state', 1)

        self.tf_trials = 5
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        queue_size = 2
        delay_btw_msgs = 0.02 #TODO: test if this value is causing any problems...
        tss = ApproximateTimeSynchronizer(
            [Subscriber(self, ImageMsg, img_topic),
             Subscriber(self, ImageMsg, depth_topic)],
             queue_size,
             delay_btw_msgs
             )
        
        tss.registerCallback(self.sense_and_act)

        self.get_logger().info('Ready to publish some twist messages!')


    def publish_state(self):
        state_msg = String()
        sub_states = [unordered for order in self.lander_sub_state_msgs for unordered in set(self.lander_sub_state) if unordered == order]
        state_msg.data = self.lander_base_state
        if len(sub_states):
            state_msg.data += "-" + "-".join(sub_states) # .split("-") to recover a list
        self.state_pub.publish(state_msg)
        self.get_logger().info(f'Current state: {state_msg.data}')


    def get_tf(self, t=0.0, timeout=1.0, map_frame="map", target_frame="flying_sensor"):
        """Only needed to grab the altitude so we can simulate 
        the altitude estimation received from a real flight controller
        """
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
        """Masks the depth image received leaving only a circle
        that approximates the UAV's safety radius projected according 
        to its current altitude
        """
        proj = math.tan(FOV/2)*altitude # [m]

        depth = self.cv_bridge.imgmsg_to_cv2(depthmsg, desired_encoding='passthrough')
        depth = np.asarray(cv2.resize(depth, 
                                      (int(self.depth_new_size*depth.shape[1]/depth.shape[0]),self.depth_new_size),
                                      cv2.INTER_AREA))
        # In CARLA the depth goes up to 1000m, but we want 
        # any value bigger than max_dist to become max_dist
        depth[depth>max_dist] = np.nan
        depth[np.isnan(depth)] = max_dist

        depth_center = depth.shape[0]//2, depth.shape[1]//2

        # self.safety_radius and proj are in metres
        # depth.shape[1] is in px
        safety_radius_pixels = int(self.safety_radius/(proj/depth.shape[1]))
        mask = np.zeros_like(depth)
        mask = cv2.circle(mask, (depth_center[1],depth_center[0]), safety_radius_pixels, 255, -1)
        depth[mask!=255] = 1000
        
        img_msg = self.cv_bridge.cv2_to_imgmsg(depth.astype('uint8'), encoding='mono8')
        self.depth_proj_pub.publish(img_msg)
        return depth[1000>depth].std(),depth[1000>depth].min()



    def error_from_semantics(self, rgbmsg, altitude, prompts):
        """Normalized XY error according to the best place to land defined by the heatmap
        received from the generate_landing_heatmap service
        The heatmap received will be a mono8 image where the higher the value 
        the better the place for landing.
        """
        proj = math.tan(FOV/2)*altitude
        self.get_logger().debug(f'Sending heatmap service request...')
        response = self.get_heatmap(rgbmsg, prompts, erosion_size=self.heatmap_mask_erosion)
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

        # Reduce the received heatmap to a size that is approximately proportional
        # to the projection of the UAV's safety radius, but with these details:
        # - Always use odd values for height and width to guarantee a pixel at the centre
        # - Limit the maximum number of pixels according to self.max_seg_height 
        # to avoid problems with noise in the semantic segmentation
        safety_radius_pixels = int(self.safety_radius/(2*proj/heatmap.shape[1]))
        resize = heatmap.shape[0]//safety_radius_pixels
        resize += 1-(resize % 2) # always odd
        resize = resize if resize < self.max_seg_height else self.max_seg_height
        resize_w = int(resize*(heatmap.shape[1]/heatmap.shape[0]))
        resize_w = resize_w + 1-(resize_w % 2) # always odd

        if self.heatmap_mov_avg is None:
            # like an RGB image so opencv can easily resize it...
            self.heatmap_mov_avg = np.zeros((resize, resize_w, self.mov_avg_size), dtype='uint8')
        
        # The resolution of the heatmap changes according to the UAV's safety projection (altitude)
        # Therefore the array used for the moving average needs to be resized as well
        self.heatmap_mov_avg = cv2.resize(self.heatmap_mov_avg,
                                            (resize_w, resize),cv2.INTER_AREA)
        heatmap_resized = cv2.resize(heatmap,
                                     (self.heatmap_mov_avg.shape[1],self.heatmap_mov_avg.shape[0]),cv2.INTER_AREA)
        
        # Add the received heatmap to the moving average array
        self.heatmap_mov_avg[...,self.mov_avg_counter] = heatmap_resized
        # Calculates the average heatmap according to the values stored in the moving average array
        heatmap_resized = self.heatmap_mov_avg.mean(axis=2).astype('uint8')
        if self.mov_avg_counter < (self.mov_avg_size-1):
            self.mov_avg_counter += 1
        else:
            self.mov_avg_counter = 0

        heatmap_center = heatmap_resized.shape[0]/2, heatmap_resized.shape[1]/2
        # The heatmap values are higher for better places (pixels) to land, 
        # but argsort will give, by default, values in ascending order.
        # Descending order, best landing candidates (image coordinates, 0,0 at the top left)
        xy_idx = np.dstack(np.unravel_index(np.argsort(heatmap_resized.ravel()), heatmap_resized.shape))[0][::-1].astype('float16')
        # Change from image coordinates to normalized coordinates in relation to the centre of the image
        xy_idx[:,0] =  (-(xy_idx[:,0] - int(heatmap_center[0]))) / heatmap_center[0]
        xy_idx[:,1] = (xy_idx[:,1] - int(heatmap_center[1])) / heatmap_center[1]

        self.get_logger().debug(f'Publishing resized heatmap image at {self.heatmap_topic}')
        img_msg = self.cv_bridge.cv2_to_imgmsg(heatmap_resized, encoding='mono8')
        self.heatmap_pub.publish(img_msg)
        return xy_idx


    def get_altitude(self):
        """Simulate the altitude read from the flight controller internal estimation
        TODO: a real UAV may use AGL (rangefinder), MSL (barometer) or a mixture
        """
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

        curr_time_sec = self.get_clock().now().nanoseconds/1E9

        delta_t_sec = curr_time_sec - self.prev_time_sec
        if delta_t_sec == 0:
            return
        self.prev_time_sec = curr_time_sec

        elapsed_time_sec = curr_time_sec-self.init_time_sec
        conservative_gain = 1-np.exp(1-self.max_landing_time_sec/elapsed_time_sec)
        self.conservative_gain = conservative_gain if conservative_gain > self.min_conservative_gain else self.min_conservative_gain
        self.get_logger().info(f'Elapsed time [s]: {elapsed_time_sec} (curr. loop freq. {1/delta_t_sec:.2f}Hz) - Conservative gain: {self.conservative_gain}')

        self.lander_base_state = None
        self.lander_sub_state = []

        altitude = self.get_altitude()
        if altitude is None:
            return
        
        self.get_logger().info(f"Altitude: {altitude:.2f}")

        x = y = z = 0.0
        if altitude<self.altitude_landed:
            self.landing_done = True
            self.lander_base_state = "LANDED"
            self.shutdown_after_landed()
        else:
            self.lander_base_state = "SEARCHING"
            if self.giveup_landing_timer == 0:
                time_since_giveup_landing = 0
            else:
                time_since_giveup_landing = (self.get_clock().now().nanoseconds/1E9 - self.giveup_landing_timer)
                self.get_logger().warn(f"Time since giveup_landing_timer enabled: {time_since_giveup_landing}")

            depth_std, depth_min = self.depth_proj(depthmsg, altitude, max_dist=self.max_depth_sensing)

            xs_err = ys_err = 0.0
            # We have two sets of prompts used to generate the landing heatmap
            # One set (PROMPTS_SEARCHING) is used when the UAV is flying above any obstacle (safety information previously known)
            # and it's searching for a place to land
            prompts = PROMPTS_SEARCHING
            if altitude < self.safe_altitude*0.8: # 0.8 is to give a margin for the heatmap generation (moving average)
                prompts = PROMPTS_DESCENDING      #TODO: fix this hack...

            xy_err = self.error_from_semantics(rgbmsg, altitude, prompts)
            xs_err = xy_err[0,0]
            ys_err = xy_err[0,1]
            self.get_logger().info(f"Current prompts: {prompts}")
            self.get_logger().info(f"Segmentation X,Y err: {xs_err:.2f},{ys_err:.2f}, Depth std, min: {depth_std:.2f},{depth_min:.2f}")
            
            x = xs_err
            y = ys_err

            # Very rudimentary filter
            x = x if abs(x) > self.zero_error_eps else 0.0
            y = y if abs(y) > self.zero_error_eps else 0.0

            zero_xy_error = (abs(x)+abs(y)) == 0.0
            # TODO: improve the flat_surface_below definition
            flat_surface_below = depth_std < self.depth_smoothness/self.conservative_gain
            # TODO: improve the no_collisions_ahead definition
            no_collisions_ahead = (depth_min >= self.max_depth_sensing) or abs(altitude - depth_min) < self.depth_smoothness
            give_up_landing_here = (time_since_giveup_landing > self.giveup_after_sec)
            self.get_logger().info(f"zero_xy_error: {zero_xy_error}, flat_surface_below: {flat_surface_below}, no_collisions_ahead: {no_collisions_ahead}, give_up_landing_here: {give_up_landing_here}")

            # altitude >= self.safe_altitude means the UAV is flying high enough and there are no obstacles to worry about
            if altitude >= self.safe_altitude:
                self.lander_sub_state.append("SAFE_ALTITUDE")
                if give_up_landing_here:
                    self.lander_base_state = "RESTARTING"
                    if self.search4new_place_timer==0:
                        # it will get a direction, random or based on the current heatmap, and move towards that
                        # direction for self.search4new_place_max_time seconds trying to find a new place to start looking
                        # for a place to land again (therefore avoiding getting stuck to current spot)
                        if self.use_random_search4new_place:
                            self.search4new_place_direction = np.random.rand(2)-1 # uniform random [-0.5,0.5] search direction
                        else:
                            # xy_err are already ordered according to the objective function 
                            # ("emptiness" and distance to the UAV) and their values are normalized in relation to the heatmap.
                            # Then it filters values at least distant 0.5 (normalized value) from the UAV and gets the first as its search direction.
                            self.search4new_place_direction = xy_err[(xy_err**2).sum(axis=1) >= 0.5][0] 
                        self.search4new_place_timer = curr_time_sec
                    search4new_place_time_passed = (self.get_clock().now().nanoseconds/1E9 - self.search4new_place_timer)
                    if search4new_place_time_passed > self.search4new_place_max_time:
                        self.giveup_landing_timer = 0 # safe place, reset giveup_landing_timer
                        self.search4new_place_timer = 0
                        self.get_logger().error(f"Search 4 new place finished!")
                    else:
                        self.get_logger().error(f"Search 4 new place is ON ({search4new_place_time_passed})!")
                    x,y = self.search4new_place_direction
                elif zero_xy_error:
                    z = -self.z_speed  
            else:
                self.get_logger().warn(f"Safe altitude breached, no movements allowed on XY!")
                # giveup_landing_timer helps filtering noisy decisions as the UAV will only give up landing 
                # on the current spot after one of the triggers consistently flags a bad place for
                # at least self.giveup_after_sec seconds
                if zero_xy_error and flat_surface_below and no_collisions_ahead and not give_up_landing_here:
                    self.giveup_landing_timer = 0 # things look fine, reset give up timer
                    z = -self.z_speed
                else:
                    # something doesn't look good, start giveup_landing_timer if it was not already started
                    if self.giveup_landing_timer == 0:
                        self.giveup_landing_timer = curr_time_sec
                    else:
                        self.lander_base_state = "RESTARTING"
                    # since things don't look good, stop descending and start ascending
                    z = self.z_speed
                x = y = 0.0 # below safe altitude, no movements allowed on XY

            if zero_xy_error:
                self.lander_sub_state.append("ZERO_XY_ERROR")
            if flat_surface_below:
                self.lander_sub_state.append("FLAT_SURFACE_BELOW")
            if no_collisions_ahead:
                self.lander_sub_state.append("NO_COLLISIONS_AHEAD")
        
        self.publish_state()

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
        #TODO: research a solution and fix service callback hack (I think Galactic is the problem... but I am not sure)

        #request.image, request.prompts, request.erosion_size
        self.req.image = image_msg
        # the service expects a string of prompts separated by ';'
        self.req.prompts = ";".join(prompts)
        self.req.erosion_size = int(erosion_size)
        self.req.safety_threshold = self.safety_threshold*float(self.conservative_gain)

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