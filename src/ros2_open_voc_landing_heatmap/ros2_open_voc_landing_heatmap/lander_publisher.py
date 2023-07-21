"""
Autonomous landing module  

Uses semantic segmentation and depth data to detect safe landing zones.
Controls UAV movement and landing approach.
"""

import sys
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

negative_prompts = ["building, house, apartment-building, warehouse, shed, garage", 
                    "roof, rooftop, terrace, shelter, dome, canopy, ceiling", 
                    "tree, bush, tall-plant", 
                    "people, crowd", 
                    "vehicle, car, train", 
                    "lamp-post, transmission-line", 
                    "fence, wall, hedge", 
                    "road, street, avenue, highway, drive, lane"]
positive_prompts = ["grass, backyard, frontyard, courtyard, lawn", 
                    "sports-field, park, open-area, open-space"] 

NEGATIVE_PROMPTS = ";".join(negative_prompts)
POSITIVE_PROMPTS = ";".join(positive_prompts)


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
    is_safe: bool = False             # UAV can move in the XY directions safely
    is_clear: bool = False            # UAV can descend safely, segmentation shows no obstacles
    is_collision_free: bool = False   # UAV can descend safely, depth shows no obstacles
    is_flat: bool = False             # UAV can land safely, ground is flat (enough)
    conservative_gain: float = 1.0
    delta_time_sec: float = 0.0
    elapsed_time_sec: float = 0.0
    altitude: float = 0.0


class LandingModule(Node):

    def __init__(self, debug=False):
        super().__init__('landing_module')
        self.declare_parameter('img_topic', '/carla/flying_sensor/rgb_down/image')
        self.declare_parameter('depth_topic', '/carla/flying_sensor/depth_down/image')
        self.declare_parameter('heatmap_topic', '/heatmap')
        self.declare_parameter('depth_proj_topic', '/depth_proj')
        self.declare_parameter('twist_topic', '/quadctrl/flying_sensor/ctrl_twist_sp')
        self.declare_parameter('beta', 1/20)
        self.declare_parameter('gain', 20)
        self.declare_parameter('z_speed', 3.0)
        self.declare_parameter('depth_smoothness', 0.5) # CARLA's values oscillate on flat surfaces
        self.declare_parameter('altitude_landed', 1)
        self.declare_parameter('safe_altitude', 50)
        self.declare_parameter('safety_radius', 2.0)
        self.declare_parameter('safety_threshold', 0.6)
        self.declare_parameter('giveup_after_sec', 5)
        self.declare_parameter('max_depth_sensing', 20)
        self.declare_parameter('use_random_search4new_place', False)
        self.declare_parameter('heatmap_mask_erosion', 2)
        self.declare_parameter('search4new_place_max_time', 20)
        self.declare_parameter('max_seg_height', 17)
        self.declare_parameter('zero_error_eps', 0.1)
        self.declare_parameter('max_landing_time_sec', 5*60)
        self.declare_parameter('min_conservative_gain', 0.1)
        self.declare_parameter('sensor_warm_up_cycles', 10)
        self.declare_parameter('negative_prompts', NEGATIVE_PROMPTS)
        self.declare_parameter('positive_prompts', POSITIVE_PROMPTS)
        img_topic = self.get_parameter('img_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        heatmap_topic = self.get_parameter('heatmap_topic').value
        depth_proj_topic = self.get_parameter('depth_proj_topic').value
        twist_topic = self.get_parameter('twist_topic').value
        self.beta = self.get_parameter('beta').value
        self.gain = self.get_parameter('gain').value
        self.z_speed = self.get_parameter('z_speed').value
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
        self.sensor_warm_up_cycles = self.get_parameter('sensor_warm_up_cycles').value
        self.negative_prompts = self.get_parameter('negative_prompts').value
        self.positive_prompts = self.get_parameter('positive_prompts').value
        self.add_on_set_parameters_callback(self.parameters_callback)

        assert self.min_conservative_gain > 0, "Min conservative time must be bigger than zero!"

        self.debug = debug

        self.cycles = 0

        self.heatmap_result = None
        self.rgbmsg = None
        self.depthmsg = None

        self.mov_avg_counter = 0

        self.heatmap_filtered = None

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
                self.get_logger().warn(f'Parameter updated: {param.name} = {param.value}')
            except AttributeError:
                return SetParametersResult(successful=False)
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
        proj = math.tan(FOV/2)*self.landing_status.altitude # [m]
        max_dist = self.max_depth_sensing

        depth = self.cv_bridge.imgmsg_to_cv2(depthmsg, desired_encoding='passthrough')

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
        depth[mask!=255] = max_dist
        depth_std = depth[mask==255].std()
        depth_min = depth[mask==255].min()
        
        img_msg = self.cv_bridge.cv2_to_imgmsg((255*depth/max_dist).astype('uint8'), encoding='mono8')
        img_msg.header.frame_id = depthmsg.header.frame_id
        self.depth_proj_pub.publish(img_msg)
        return depth_std, depth_min



    def get_xy_error_from_semantics(self, heatmap_msg):
        """Normalized XY error according to the best place to land defined by the heatmap
        received from the generate_landing_heatmap service
        The heatmap received will be a mono8 image where the higher the value 
        the better the place for landing.
        """
        heatmap = self.cv_bridge.imgmsg_to_cv2(heatmap_msg, desired_encoding='mono8')
       
        # Debug
        # heatmap = np.zeros(heatmap.shape, dtype=heatmap.dtype)
        # heatmap[:50,:50] = 255
        # heatmap[-50:,-50:] = 255

        # Reduce the received heatmap size
        resize_h = 100
        resize_w = int(resize_h*(heatmap.shape[1]/heatmap.shape[0]))
        resize_w = resize_w + 1-(resize_w % 2) # always odd

        if self.heatmap_filtered is None:
            self.heatmap_filtered = np.zeros((resize_h, resize_w), dtype=float)
        
        heatmap_resized = cv2.resize(heatmap,
                                     (self.heatmap_filtered.shape[1],self.heatmap_filtered.shape[0]),cv2.INTER_AREA)

        # Add the received heatmap to the moving average array
        self.heatmap_filtered += self.beta*(heatmap_resized-self.heatmap_filtered)
        heatmap_resized = self.heatmap_filtered.astype('uint8')

        heatmap_center = np.asarray([heatmap_resized.shape[0]/2, heatmap_resized.shape[1]/2])
        
        # add a black border to avoid problems with distanceTransform
        heatmap_resized[:,0] = 0
        heatmap_resized[:,-1] = 0
        heatmap_resized[0,:] = 0
        heatmap_resized[-1,:] = 0
        
        _,tmp_thresh = cv2.threshold(heatmap_resized,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        heatmap_dist_function = cv2.distanceTransform(tmp_thresh, cv2.DIST_L2, 5)
        cv2.normalize(heatmap_dist_function, heatmap_dist_function, 0, 1.0, cv2.NORM_MINMAX)
       
        # Check area, perimeter, distance from center
        _, dist_thrs = cv2.threshold(heatmap_dist_function, 0.6, 1.0, cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(dist_thrs.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xy_idx = [[1.0,0]] # if nothing is found below, move forward
        objective_values = [0.0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if (area == 0) or (perimeter == 0):
                continue
            M = cv2.moments(cnt)
            cy = int(M['m10']/M['m00'])
            cx = int(M['m01']/M['m00'])
            d_center = np.sqrt(((heatmap_center-[cx,cy])**2).sum())
            objective = (1/perimeter)*area/(d_center+1) # complex shapes will have longer perimeter
            objective_values.append(objective)
            x = (heatmap_center[0] - cx)/heatmap_resized.shape[0]
            y = -(heatmap_center[1] - cy)/heatmap_resized.shape[1]
            xy_idx.append([x,y])
        xy_idx = np.asarray(xy_idx)
        desc_order = np.argsort(objective_values)[::-1]
        xy_idx = xy_idx[desc_order]

        yc = int(-(xy_idx[0,0]*heatmap_resized.shape[0]-heatmap_center[0]))
        xc = int(xy_idx[0,1]*heatmap_resized.shape[1]+heatmap_center[1])
        img_msg = self.cv_bridge.cv2_to_imgmsg(cv2.circle(cv2.cvtColor((dist_thrs*255).astype('uint8'), cv2.COLOR_GRAY2BGR), (xc,yc), 3, (0,0,255), -1))
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
    

    def publish_twist(self, x, y, z):
        twist = Twist()
        # The UAV's maximum bank angle is limited to a very small value
        # and this is why such a simple control works.
        # Additionally, the assumption is that the maximum speed is very low
        # otherwise the moving average used in the semantic segmentation will break.
        twist.linear.x = float(x * self.gain)  # the max bank angle is limited (tiltMax), therefore the gain is here to saturate
        twist.linear.y = float(-y * self.gain)
        twist.linear.z = float(z)
        
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        if not self.debug:
            self.get_logger().info(f'Publishing velocities ({(twist.linear.x, twist.linear.y, twist.linear.z)})')
            self.twist_pub.publish(twist)
        else:
            self.get_logger().error("Debug mode active: no velocities published!")


    def publish_status(self):
        state_msg = String() # easy to break apart without the need for a custom message...
        msg_str = str(self.landing_status.state).split('.')[1]
        self.get_logger().info(f'Current state: {msg_str}')
        if self.landing_status.is_safe:
            self.get_logger().info("Safe altitude")
            msg_str += "-safe"
        else:
            self.get_logger().warn(f"Safe altitude breached, no movements allowed on XY!")
        if self.landing_status.is_clear:
            self.get_logger().info("Segmentation clear")
            msg_str += "-seg_clear"
        else:
            self.get_logger().warn(f"Segmentation detected obstacle")
        if self.landing_status.is_collision_free:
            self.get_logger().info("Depth collision free")
            msg_str += "-depth_clear"
        else:
            self.get_logger().warn(f"Depth collision detected")
        if self.landing_status.is_flat:
            self.get_logger().info("Flat ground")
            msg_str += "-flat"
        else:
            self.get_logger().warn(f"Bumpy ground")

        msg_str += f"-ALT:{self.landing_status.altitude:.3f}"
        self.get_logger().info(f"Altitude: {self.landing_status.altitude:0.3f} m")
        msg_str += f"-CSG:{self.landing_status.conservative_gain:0.3f}"
        self.get_logger().info(f"Conservative Gain: {self.landing_status.conservative_gain:0.3f}")
        msg_str += f"-DTS:{self.landing_status.delta_time_sec:0.3f}"
        self.get_logger().info(f"Loop Freq.: {1/self.landing_status.delta_time_sec:0.3f} Hz")
        msg_str += f"-ETS:{self.landing_status.elapsed_time_sec:0.3f}"
        self.get_logger().info(f"Elapsed Time: {self.landing_status.elapsed_time_sec:0.3f} s")
        state_msg.data = msg_str
        self.state_pub.publish(state_msg)


    def sense_and_act(self, rgbmsg, depthmsg):
        if self.heatmap_result is None:
            # Beware: spaghetti-code state machine below!
            curr_time_sec = self.get_clock().now().nanoseconds/1E9
            self.landing_status.delta_time_sec = curr_time_sec - self.prev_time_sec
            if self.landing_status.delta_time_sec == 0:
                return
            self.prev_time_sec = curr_time_sec
            self.landing_status.elapsed_time_sec = curr_time_sec-self.init_time_sec

            self.landing_status.altitude = self.get_altitude()
            if self.landing_status.altitude is None:
                return
            
            if self.landing_status.altitude >= self.safe_altitude:
                self.landing_status.is_safe = True
            else:
                self.landing_status.is_safe = False

            negative_prompts = self.negative_prompts
            positive_prompts = self.positive_prompts

            # The conservative_gain is a very simple (hacky?) way to force the system to relax its decisions as time passes 
            # because at the end of the day it will be limited by its battery and the worst scenario is to fall from the sky
            # - flatness (is_flat_dynamic_decision)
            # - minimum distance to obstacles (is_collision_free_dynamic_decision)
            # - maximum acceptable heatmap location error before switching to landing (is_clear_dynamic_decision)
            conservative_gain = 1-np.exp(1-self.max_landing_time_sec/self.landing_status.elapsed_time_sec)
            self.landing_status.conservative_gain = conservative_gain if conservative_gain > self.min_conservative_gain else self.min_conservative_gain
        
            depth_std, depth_min = self.get_depth_stats(depthmsg)
            self.get_logger().info(f"Depth STD, MIN: {depth_std:.2f},{depth_min:.2f}")


            #TODO: research a solution to sync service call and received messages better than this...
            #request.image, request.positive_prompts, request.negative_prompts, request.erosion_size
            self.req.image = rgbmsg
            # the service expects a string of prompts separated by ';'
            self.req.positive_prompts = positive_prompts
            self.req.negative_prompts = negative_prompts
            self.req.safety_threshold = self.safety_threshold

            def future_done_callback(future):
                heatmap_msg = future.result().heatmap
                x,y,z = self.state_update(curr_time_sec, heatmap_msg, depth_std, depth_min)
                self.publish_status()
                self.publish_twist(x,y,z)
                self.heatmap_result = None
                if self.landing_status.state == LandingState.LANDED:
                    exit(0)

            self.heatmap_result = self.cli.call_async(self.req)
            self.heatmap_result.add_done_callback(future_done_callback)

            return


    def state_update(self, curr_time_sec, heatmap_msg, depth_std, depth_min):
        estimated_travelled_distance = self.z_speed*self.landing_status.delta_time_sec # TODO:improve this estimation or add some extra margin
        altitude_landed_dynamic = self.altitude_landed if self.altitude_landed > estimated_travelled_distance else estimated_travelled_distance
        landed_trigger = self.landing_status.altitude <= altitude_landed_dynamic
        xy_err = self.get_xy_error_from_semantics(heatmap_msg)
        xs_err, ys_err = xy_err[0]
        self.get_logger().info(f"Segmentation X,Y ERR: {xs_err:.2f},{ys_err:.2f}")
        # Very rudimentary filter
        xs_err_filtered = xs_err if abs(xs_err) > self.zero_error_eps else 0.0
        ys_err_filtered = ys_err if abs(ys_err) > self.zero_error_eps else 0.0
        self.landing_status.is_clear = (abs(xs_err_filtered)+abs(ys_err_filtered)) == 0.0
        xs_err_filtered_dynamic = xs_err if abs(xs_err) > self.zero_error_eps/self.landing_status.conservative_gain else 0.0
        ys_err_filtered_dynamic = ys_err if abs(ys_err) > self.zero_error_eps/self.landing_status.conservative_gain else 0.0
        is_clear_dynamic_decision = (abs(xs_err_filtered_dynamic)+abs(ys_err_filtered_dynamic)) == 0.0
        # TODO: improve the flat surface definition
        self.landing_status.is_flat = depth_std < self.depth_smoothness
        is_flat_dynamic_decision = depth_std < self.depth_smoothness/self.landing_status.conservative_gain
        # TODO: improve the is_collision_free definition
        self.landing_status.is_collision_free = (depth_min >= self.max_depth_sensing) # sensor saturated
        safety_distance = self.safety_radius if self.safety_radius > estimated_travelled_distance else estimated_travelled_distance
        self.landing_status.is_collision_free |= depth_min >= safety_distance
        self.landing_status.is_collision_free |= self.landing_status.altitude <= safety_distance # at this point the flatness is the only depth-based defense...
        is_collision_free_dynamic_decision = (depth_min >= self.max_depth_sensing)
        safety_radius_dynamic_decision = self.safety_radius*self.landing_status.conservative_gain
        safety_radius_dynamic_decision = safety_radius_dynamic_decision if safety_radius_dynamic_decision >= estimated_travelled_distance else estimated_travelled_distance
        is_collision_free_dynamic_decision |= depth_min >= safety_radius_dynamic_decision
        is_collision_free_dynamic_decision |= self.landing_status.altitude <= safety_radius_dynamic_decision

        # Trying to isolate all the sensing above 
        # and the state switching decisions below.
        # TODO: isolate the sensing and state decision in two distinct methods...
        if landed_trigger:
            self.landing_status.state = LandingState.LANDED
        elif ~np.isfinite(xs_err) or ~np.isfinite(ys_err) or ~np.isfinite(depth_std) or ~np.isfinite(depth_min) or self.cycles < self.sensor_warm_up_cycles:
            self.landing_status.state = LandingState.SENSOR_ERROR
            self.giveup_landing_timer = 0
            self.cycles += 1
        elif self.giveup_landing_timer == 0:
            if is_clear_dynamic_decision and is_collision_free_dynamic_decision and is_flat_dynamic_decision:
                self.landing_status.state = LandingState.LANDING
            elif self.landing_status.is_safe:
                self.landing_status.state = LandingState.SEARCHING
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
                            self.search4new_place_direction = xy_err[1] if len(xy_err)>1 else xy_err[0] # get the next in line...
                        self.search4new_place_timer = curr_time_sec
                    search4new_place_time_passed = (curr_time_sec - self.search4new_place_timer)
                    if search4new_place_time_passed > self.search4new_place_max_time*self.landing_status.conservative_gain:
                        self.giveup_landing_timer = 0 # safe place, reset giveup_landing_timer
                        self.search4new_place_timer = 0
                        self.landing_status.state = LandingState.SEARCHING
                else:
                    self.landing_status.state = LandingState.CLIMBING          
            elif is_clear_dynamic_decision and is_collision_free_dynamic_decision and is_flat_dynamic_decision:
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
    debug = False
    if len(sys.argv)>1:
        if "debug" in sys.argv[1]:
            debug = True
    rclpy.init()
    landing_module = LandingModule(debug)
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