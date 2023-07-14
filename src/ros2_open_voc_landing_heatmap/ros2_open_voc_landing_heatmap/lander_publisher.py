"""
Autonomous landing module  

Uses semantic segmentation and depth data to detect safe landing zones.
Controls UAV movement and landing approach.
"""

import math
from enum import Enum
from dataclasses import dataclass

import numpy as np
import cv2


from ros2_open_voc_landing_heatmap_srv.srv import GetLandingHeatmap
from std_msgs.msg import String
from sensor_msgs.msg import Image as ImageMsg
from geometry_msgs.msg import Twist

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.time import Time, Duration
from rcl_interfaces.msg import SetParametersResult

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


from message_filters import ApproximateTimeSynchronizer, Subscriber


from cv_bridge import CvBridge


# We use the same FOV from the camera in the stereo pair for the 
# semantic sementation because the second doesn't need a precise projection
FOV = math.radians(73) #TODO: get this from the camera topic...

NEGATIVE_PROMPTS_SEARCHING = ["building", "house", "roof", "asphalt", "tree", "road", "water", "wall", "fence", "transmission lines", "lamp post", "vehicle", "people"]
POSITIVE_PROMPTS_SEARCHING = ["grass", "field", "sand"]
NEGATIVE_PROMPTS_LANDING= ["vehicle", "people"]
POSITIVE_PROMPTS_LANDING= ["grass", "field", "sand"]


class LandingState(Enum):
    SEARCHING = 0 
    LANDING = 1
    WAITING = 2
    CLIMBING = 3
    RESTARTING = 4
    LANDED = 5
    SHUTTING_DOWN = 6
    SENSOR_ERROR = 7

@dataclass
class LandingStatus:
    state: LandingState = LandingState.SEARCHING
    is_safe: bool = False   # UAV can move in the XY directions safely
    is_clear: bool = False  # UAV can descend safely, no obstacles
    is_flat: bool = False   # UAV can land safely, ground is flat (enough)

class LandingModule(Node):

    def __init__(self):
        super().__init__('landing_module')
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
        self.declare_parameter('safety_threshold', 0.5)
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
        heatmap_topic = self.get_parameter('heatmap_topic').value
        depth_proj_topic = self.get_parameter('depth_proj_topic').value
        twist_topic = self.get_parameter('twist_topic').value
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
        self.add_on_set_parameters_callback(self.parameters_callback)

        self.heatmap_result = None
        self.rgbmsg = None
        self.depthmsg = None

        self.mov_avg_counter = 0

        self.heatmap_mov_avg = None

        self.img_msg = None

        self.giveup_landing_timer = 0

        self.search4new_place_timer = 0

        self.search4new_place_direction = (0,0)

        self.init_time_sec = self.get_clock().now().nanoseconds/1E9
        self.prev_time_sec = self.get_clock().now().nanoseconds/1E9

        self.landing_status = LandingStatus()

        self.cli = self.create_client(GetLandingHeatmap, 'generate_landing_heatmap',
                                      callback_group=ReentrantCallbackGroup())
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('generate_landing_heatmap service not available, waiting again...')
        self.req = GetLandingHeatmap.Request()
        self.cv_bridge = CvBridge()
                
        self.twist_pub = self.create_publisher(Twist, twist_topic,1)
        self.heatmap_pub = self.create_publisher(ImageMsg, heatmap_topic,1)
        self.depth_proj_pub = self.create_publisher(ImageMsg, depth_proj_topic,1)
        self.state_pub = self.create_publisher(String, 'lander_state', 1)

        self.tf_trials = 5
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        queue_size = 1
        delay_btw_msgs = 0.02 #TODO: test if this value is causing any problems...
        tss = ApproximateTimeSynchronizer(
            [Subscriber(self, ImageMsg, img_topic),
             Subscriber(self, ImageMsg, depth_topic)],
             queue_size,
             delay_btw_msgs
             )
        
        tss.registerCallback(self.sense_and_act)

        self.get_logger().info('Ready to publish some twist messages!')


    def parameters_callback(self, params):
        for param in params:
            try:
                var_type = type(getattr(self, param.name))
                setattr(self, param.name, var_type(param.value))
                self.get_logger().info(f'Parameter updated: {param.name} = {param.value}')
            except AttributeError:
                print("ok - AttributeError")
                return SetParametersResult(successful=False)
        print("ok")
        return SetParametersResult(successful=True)
    

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


    def get_depth_stats(self, depthmsg):
        """Masks the depth image received leaving only a circle
        that approximates the UAV's safety radius projected according 
        to its current altitude
        """
        proj = math.tan(FOV/2)*self.curr_altitude # [m]
        max_dist = self.max_depth_sensing

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
        img_msg.header.frame_id = depthmsg.header.frame_id
        self.depth_proj_pub.publish(img_msg)
        return depth[1000>depth].std(),depth[1000>depth].min()



    def get_xy_error_from_semantics(self, heatmap_msg):
        """Normalized XY error according to the best place to land defined by the heatmap
        received from the generate_landing_heatmap service
        The heatmap received will be a mono8 image where the higher the value 
        the better the place for landing.
        """
        proj = math.tan(FOV/2)*self.curr_altitude
        heatmap = 2*(self.cv_bridge.imgmsg_to_cv2(heatmap_msg, desired_encoding='mono8')/255)-1
       
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
            self.heatmap_mov_avg = np.zeros((resize, resize_w, self.mov_avg_size), dtype=float)
        
        # The resolution of the heatmap changes according to the UAV's safety projection (altitude)
        # Therefore the array used for the moving average needs to be resized as well
        self.heatmap_mov_avg = cv2.resize(self.heatmap_mov_avg,
                                            (resize_w, resize),cv2.INTER_AREA)
        heatmap_resized = cv2.resize(heatmap,
                                     (self.heatmap_mov_avg.shape[1],self.heatmap_mov_avg.shape[0]),cv2.INTER_AREA)
        
        # Add the received heatmap to the moving average array
        self.heatmap_mov_avg[...,self.mov_avg_counter] = heatmap_resized
        # Calculates the average heatmap according to the values stored in the moving average array
        heatmap_resized = (255*(self.heatmap_mov_avg.mean(axis=2)+1)/2).astype('uint8')
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
        img_msg = self.cv_bridge.cv2_to_imgmsg(heatmap_resized, encoding='mono8')
        img_msg.header.frame_id = heatmap_msg.header.frame_id
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
    

    def publish_twist(self, x, y, z, debug=False):
        twist = Twist()
        # The UAV's maximum bank angle is limited to a very small value
        # and this is why such a simple control works.
        # Additionally, the assumption is that the maximum speed is very low
        # otherwise the moving average used in the semantic segmentation will break.
        twist.linear.x = float(x * self.gain)
        twist.linear.y = float(-y * self.gain)
        twist.linear.z = float(z)
        
        if debug:
            twist.linear.x = twist.linear.y = twist.linear.z = 0.0 ##DEBUG
        
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        self.get_logger().info(f'Publishing velocities ({(twist.linear.x, twist.linear.y, twist.linear.z)})')
        self.twist_pub.publish(twist)


    def publish_status(self):
        state_msg = String()
        msg_str = str(self.landing_status.state).split('.')[1]
        if self.landing_status.is_safe:
            msg_str += "-safe"
        if self.landing_status.is_clear:
            msg_str += "-clear"
        if self.landing_status.is_flat:
            msg_str += "-flat"
        state_msg.data = msg_str
        self.state_pub.publish(state_msg)
        self.get_logger().info(f'Current state: {state_msg.data}')
        self.get_logger().info(f"Altitude: {self.curr_altitude:.2f}")


    def sense_and_act(self, rgbmsg, depthmsg):
        if self.heatmap_result is None:
            # Beware: spaghetti-code state machine below!
            curr_time_sec = self.get_clock().now().nanoseconds/1E9
            delta_t_sec = curr_time_sec - self.prev_time_sec
            if delta_t_sec == 0:
                return
            self.prev_time_sec = curr_time_sec
            elapsed_time_sec = curr_time_sec-self.init_time_sec

            self.curr_altitude = self.get_altitude()
            if self.curr_altitude is None:
                return
            
            if self.curr_altitude >= self.safe_altitude:
                self.landing_status.is_safe = True
            else:
                self.landing_status.is_safe = False
                self.get_logger().warn(f"Safe altitude breached, no movements allowed on XY!")

            # We have two sets of prompts used to generate the landing heatmap
            # One set (PROMPTS_SEARCHING) is used when the UAV is flying above any obstacle (safety information previously known)
            # and it's searching for a place to land
            # The multiplier 0.8 is to give a margin for the heatmap generation (moving average), mostly on its way up
            # TODO: fix this multiplier hack...
            if self.curr_altitude < self.safe_altitude*0.8:
                negative_prompts = NEGATIVE_PROMPTS_LANDING
                positive_prompts = POSITIVE_PROMPTS_LANDING
            else:
                negative_prompts = NEGATIVE_PROMPTS_SEARCHING
                positive_prompts = POSITIVE_PROMPTS_SEARCHING
            self.get_logger().info(f"Current prompts: {negative_prompts}, {positive_prompts}")

            # The conservative_gain is a very simple (hacky?) way to force the system to relax its decisions as time passes 
            # because at the end of the day it will be limited by its battery and the worst scenario is to fall from the sky
            conservative_gain = 1-np.exp(1-self.max_landing_time_sec/elapsed_time_sec)
            self.conservative_gain = conservative_gain if conservative_gain > self.min_conservative_gain else self.min_conservative_gain
            self.get_logger().info(f'Elapsed time [s]: {elapsed_time_sec} (curr. loop freq. {1/delta_t_sec:.2f}Hz) - Conservative gain: {self.conservative_gain}')
        
            depth_std, depth_min = self.get_depth_stats(depthmsg)
            self.get_logger().info(f"Depth STD, MIN: {depth_std:.2f},{depth_min:.2f}")


            #TODO: research a solution to sync service call and received messages better than this...
            #request.image, request.positive_prompts, request.negative_prompts, request.erosion_size
            self.req.image = rgbmsg
            # the service expects a string of prompts separated by ';'
            self.req.positive_prompts = ";".join(positive_prompts)
            self.req.negative_prompts = ";".join(negative_prompts)
            self.req.erosion_size = int(self.heatmap_mask_erosion)
            self.req.safety_threshold = self.safety_threshold*float(self.conservative_gain)

            def future_done_callback(future):
                heatmap_msg = future.result().heatmap
                x,y,z = self.state_update(curr_time_sec, heatmap_msg, depth_std, depth_min)
                self.publish_status()
                self.publish_twist(x,y,z)
                self.heatmap_result = None

            self.heatmap_result = self.cli.call_async(self.req)
            self.heatmap_result.add_done_callback(future_done_callback)

            return


    def state_update(self, curr_time_sec, heatmap_msg, depth_std, depth_min):
        xy_err = self.get_xy_error_from_semantics(heatmap_msg)
        xs_err = xy_err[0,0]
        ys_err = xy_err[0,1]
        self.get_logger().info(f"Segmentation X,Y ERR: {xs_err:.2f},{ys_err:.2f}")
        # Very rudimentary filter
        xs_err = xs_err if abs(xs_err) > self.zero_error_eps else 0.0
        ys_err = ys_err if abs(ys_err) > self.zero_error_eps else 0.0
        zero_xy_error = (abs(xs_err)+abs(ys_err)) == 0.0
        # TODO: improve the flat surface definition
        self.landing_status.is_flat = depth_std < self.depth_smoothness/self.conservative_gain
        # TODO: improve the no_collisions_ahead definition
        no_collisions_ahead = (depth_min >= self.max_depth_sensing) or abs(self.curr_altitude - depth_min) < self.altitude_landed
        self.landing_status.is_clear = zero_xy_error and no_collisions_ahead
        self.get_logger().info(f"zero_xy_error: {zero_xy_error}, flat_surface_below: {self.landing_status.is_flat}, no_collisions_ahead: {no_collisions_ahead}")

        # Trying to isolate all the sensing above 
        # and the state switching decisions below.
        # TODO: isolate the sensing and state decision in two distinct methods...
        if self.curr_altitude<self.altitude_landed:
            self.landing_status.state = LandingState.LANDED
        elif ~np.isfinite(xs_err) or ~np.isfinite(ys_err) or ~np.isfinite(depth_std) or ~np.isfinite(depth_min):
            self.landing_status.state = LandingState.SENSOR_ERROR
        elif self.giveup_landing_timer == 0:
            if self.landing_status.is_clear and self.landing_status.is_flat:
                self.landing_status.state = LandingState.LANDING
            elif self.landing_status.state != LandingState.SEARCHING:
                self.giveup_landing_timer = curr_time_sec
                self.landing_status.state = LandingState.WAITING
        else:
            time_since_giveup_landing = (curr_time_sec - self.giveup_landing_timer)
            self.get_logger().warn(f"Time since giveup_landing_timer enabled: {time_since_giveup_landing}")        
            if (time_since_giveup_landing > self.giveup_after_sec):
                if self.landing_status.is_safe:
                    self.landing_status.state = LandingState.RESTARTING
                    if self.search4new_place_timer == 0:
                        if self.use_random_search4new_place:
                            self.search4new_place_direction = np.random.rand(2)-1 # uniform random [-0.5,0.5] search direction
                        else:
                            # xy_err are already ordered according to the objective function 
                            # ("emptiness" and distance to the UAV) and their values are normalized in relation to the heatmap.
                            # Then it filters values at least distant 0.5 (normalized value) from the UAV and gets the first as its search direction.
                            self.search4new_place_direction = xy_err[(xy_err**2).sum(axis=1) >= 0.5][0] 
                        self.search4new_place_timer = curr_time_sec
                    search4new_place_time_passed = (curr_time_sec - self.search4new_place_timer)
                    if search4new_place_time_passed > self.search4new_place_max_time:
                        self.giveup_landing_timer = 0 # safe place, reset giveup_landing_timer
                        self.search4new_place_timer = 0
                        self.landing_status.state = LandingState.SEARCHING
                else:
                    self.landing_status.state = LandingState.CLIMBING          
            elif self.landing_status.is_clear and self.landing_status.is_flat:      
                self.landing_status.state = LandingState.LANDING
                self.giveup_landing_timer = 0

        if self.landing_status.state == LandingState.LANDED:
            x = y = z = 0.0
        elif self.landing_status.state == LandingState.SENSOR_ERROR:
            x = y = z = 0.0
        elif self.landing_status.state == LandingState.SEARCHING:
            x = xs_err
            y = ys_err
            z = 0.0
        elif self.landing_status.state == LandingState.LANDING:
            x = y = 0.0
            z = -self.z_speed
        elif self.landing_status.state == LandingState.CLIMBING:
            x = y = 0.0
            z = self.z_speed
        elif self.landing_status.state == LandingState.WAITING:
            x = y = z = 0.0
        elif self.landing_status.state == LandingState.RESTARTING:
            x,y = self.search4new_place_direction
            z = 0.0

        return x,y,z


    def on_shutdown_cb(self):
        self.landing_status.state = LandingState.SHUTTING_DOWN
        self.publish_twist(0,0,0)
        self.publish_status()
        self.get_logger().error('Shutting down... sending zero velocities!')


def main():
    rclpy.init()
    landing_module = LandingModule()
    try:
        rclpy.spin(landing_module)
    except KeyboardInterrupt:
        pass
    finally:
        landing_module.on_shutdown_cb()
        rclpy.shutdown()

    # executor = MultiThreadedExecutor(num_threads=2)
    # executor.add_node(landing_module)
    # try:
    #     executor.spin()

    # except KeyboardInterrupt:
    #     pass

    # finally:
    #     landing_module.on_shutdown_cb()
    #     executor.shutdown()
    #     landing_module.destroy_node()


if __name__ == '__main__':
    main()