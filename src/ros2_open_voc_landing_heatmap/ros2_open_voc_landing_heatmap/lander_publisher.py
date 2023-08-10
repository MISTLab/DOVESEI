"""
Autonomous landing module  

Uses semantic segmentation and depth data to detect safe landing zones.
Controls UAV movement and landing approach.
"""

import sys
import math
from enum import Enum
from dataclasses import dataclass
import json

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
                    "tree, bare tree, tree during autumn, bush, tall-plant", 
                    "water, lake, river, swimming pool",
                    "people, crowd", 
                    "vehicle, car, train", 
                    "lamp-post, transmission-line", 
                    "fence, wall, hedgerow", 
                    "road, street, avenue, highway, drive, lane",
                    "stairs, steps, footsteps"]
positive_prompts = ["grass, dead grass, backyard, frontyard, courtyard, lawn", 
                    "sports-field, park, open-area, open-space, agricultural land",
                    "parking lot, sidewalk, gravel, dirt, sand, concrete floor, asphalt"] 

PROMPT_ENGINEERING = "a bird's eye view of a {}, ingame screen shot, bad graphics, shade, shadows"

NEGATIVE_PROMPTS = ";".join(negative_prompts)
POSITIVE_PROMPTS = ";".join(positive_prompts)


class LandingState(Enum):
    SEARCHING = 0 
    AIMING = 1
    LANDING = 2
    WAITING = 3
    CLIMBING = 4
    RESTARTING = 5
    LANDED = 6
    SHUTTING_DOWN = 7
    SENSOR_ERROR = 8

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
    curr_threshold: float = 0.0
    success: bool = False


class LandingModule(Node):

    def __init__(self, debug=False, savefile=None):
        super().__init__('landing_module')
        self.declare_parameter('img_topic', '/carla/flying_sensor/rgb_down/image')
        self.declare_parameter('depth_topic', '/carla/flying_sensor/depth_down/image')
        self.declare_parameter('heatmap_topic', '/heatmap')
        self.declare_parameter('depth_proj_topic', '/depth_proj')
        self.declare_parameter('twist_topic', '/quadctrl/flying_sensor/ctrl_twist_sp')
        self.declare_parameter('beta', 1/20)
        self.declare_parameter('gain', 0.5)
        self.declare_parameter('aiming_gain_mult', 0.5)
        self.declare_parameter('z_speed_landing', 3.0)
        self.declare_parameter('z_min_speed_landing', 0.5)
        self.declare_parameter('z_gain_landing', 0.02)
        self.declare_parameter('z_speed_climbing', 6.0)
        self.declare_parameter('depth_smoothness', 0.5) # CARLA's values oscillate on flat surfaces
        self.declare_parameter('depth_decimation_factor', 10)
        self.declare_parameter('altitude_landed', 1.5)
        self.declare_parameter('safe_altitude', 50)
        self.declare_parameter('safety_radius', 1.5)
        self.declare_parameter('safety_threshold', 0.8)
        self.declare_parameter('dist_func_threshold', 0.6)
        self.declare_parameter('giveup_after_sec', 5)
        self.declare_parameter('max_depth_sensing', 20)
        self.declare_parameter('use_random_search4new_place', False)
        self.declare_parameter('search4new_place_max_time', 60)
        self.declare_parameter('max_landing_time_sec', 5*60)
        self.declare_parameter('min_conservative_gain', 0.5)
        self.declare_parameter('sensor_warm_up_cycles', 5)
        self.declare_parameter('negative_prompts', NEGATIVE_PROMPTS)
        self.declare_parameter('positive_prompts', POSITIVE_PROMPTS)
        self.declare_parameter('blur_kernel_size', 15)
        self.declare_parameter('seg_dynamic_threshold',0.10)
        self.declare_parameter('prompt_engineering', PROMPT_ENGINEERING)
        self.declare_parameter('aiming_descending_mult', 0.5)

        img_topic = self.get_parameter('img_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        heatmap_topic = self.get_parameter('heatmap_topic').value
        depth_proj_topic = self.get_parameter('depth_proj_topic').value
        twist_topic = self.get_parameter('twist_topic').value
        self.beta = self.get_parameter('beta').value
        self.input_gain = self.get_parameter('gain').value
        self.aiming_gain_mult = self.get_parameter('aiming_gain_mult').value
        self.z_speed_landing = self.get_parameter('z_speed_landing').value
        self.z_min_speed_landing = self.get_parameter('z_min_speed_landing').value
        self.z_gain_landing = self.get_parameter('z_gain_landing').value
        self.z_speed_climbing = self.get_parameter('z_speed_climbing').value
        self.depth_smoothness = self.get_parameter('depth_smoothness').value
        self.depth_decimation_factor = self.get_parameter('depth_decimation_factor').value
        self.altitude_landed = self.get_parameter('altitude_landed').value
        self.safe_altitude = self.get_parameter('safe_altitude').value
        self.safety_radius = self.get_parameter('safety_radius').value
        self.safety_threshold = self.get_parameter('safety_threshold').value
        self.dist_func_threshold = self.get_parameter('dist_func_threshold').value
        self.giveup_after_sec = self.get_parameter('giveup_after_sec').value
        self.max_depth_sensing = self.get_parameter('max_depth_sensing').value
        self.use_random_search4new_place = self.get_parameter('use_random_search4new_place').value
        self.search4new_place_max_time = self.get_parameter('search4new_place_max_time').value
        self.max_landing_time_sec = self.get_parameter('max_landing_time_sec').value
        self.min_conservative_gain = self.get_parameter('min_conservative_gain').value
        self.sensor_warm_up_cycles = self.get_parameter('sensor_warm_up_cycles').value
        self.negative_prompts = self.get_parameter('negative_prompts').value
        self.positive_prompts = self.get_parameter('positive_prompts').value
        self.blur_kernel_size = self.get_parameter('blur_kernel_size').value
        self.prompt_engineering = self.get_parameter('prompt_engineering').value
        self.seg_dynamic_threshold = self.get_parameter('seg_dynamic_threshold').value
        self.aiming_descending_mult = self.get_parameter('aiming_descending_mult').value
        self.add_on_set_parameters_callback(self.parameters_callback)

        if not self.min_conservative_gain > 0:
            self.get_logger().error("Min conservative time must be bigger than zero!")
            exit(1)
        if not self.altitude_landed >= self.safety_radius:
            self.get_logger().error("It will never land if self.altitude_landed < self.safety_radius :(")
            exit(1)

        self.z_speed = self.z_speed_landing
        self.gain = self.input_gain
        self.debug = debug
        self.savefile = savefile
        self.savedict = { }
        self.curr_params = {
                        "beta": self.beta,
                        "gain": self.gain,
                        "z_speed_landing": self.z_speed_landing,
                        "z_min_speed_landing": self.z_min_speed_landing,
                        "z_gain_landing": self.z_gain_landing,
                        "z_speed_climbing": self.z_speed_climbing,
                        "depth_smoothnes": self.depth_smoothness,
                        "depth_decimation_factor": self.depth_decimation_factor,
                        "altitude_landed": self.altitude_landed,
                        "safe_altitude": self.safe_altitude,
                        "safety_radius": self.safety_radius,
                        "safety_threshold": self.safety_threshold,
                        "dist_func_threshold": self.dist_func_threshold,
                        "giveup_after_sec": self.giveup_after_sec,
                        "max_depth_sensing": self.max_depth_sensing,
                        "use_random_search4new_place": self.use_random_search4new_place,
                        "search4new_place_max_time": self.search4new_place_max_time,
                        "max_landing_time_sec": self.max_landing_time_sec,
                        "min_conservative_gain": self.min_conservative_gain,
                        "sensor_warm_up_cycles": self.sensor_warm_up_cycles,
                        "negative_prompts": self.negative_prompts,
                        "positive_prompts": self.positive_prompts,
                        "blur_kernel_size": self.blur_kernel_size,
                        "prompt_engineering": self.prompt_engineering
                        }
        self.savedict[0] = self.curr_params

        self.cycles = 0
        
        self.proj = 0

        self.heatmap_result = None
        self.rgbmsg = None
        self.depthmsg = None

        self.mov_avg_counter = 0

        self.heatmap_filtered = None
        self.focus_mask_radius = None

        self.img_msg = None

        self.giveup_landing_timer = 0

        self.search4new_place_timer = 0

        self.search4new_place_direction = (0,0)

        self.int_x = 0.0
        self.int_y = 0.0
        self.int_x_sat = self.int_y_sat = 1.0

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

        self.init_time_sec = self.get_clock().now().nanoseconds/1E9
        self.prev_time_sec = self.get_clock().now().nanoseconds/1E9
        tss.registerCallback(self.sense_and_act)

        self.get_logger().info('Ready to publish some twist messages!')


    def parameters_callback(self, params):
        for param in params:
            try:
                var_type = type(getattr(self, param.name))
                setattr(self, param.name, var_type(param.value))
                self.curr_params[param.name] = param.value
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
        max_dist = self.max_depth_sensing

        depth = self.cv_bridge.imgmsg_to_cv2(depthmsg, desired_encoding='passthrough')

        # In CARLA the depth goes up to 1000m, but we want 
        # any value bigger than max_dist to become max_dist
        depth[depth>max_dist] = max_dist
        
        # To make sure we won't ignore those points
        depth[np.logical_not(np.isfinite(depth))] = 0.0

        # The values depth_std and depth_min are used for flatness and collision detection,
        # therefore they can't ignore the "holes" in the calculated disparity of real sensors.
        # By resising with cv2.INTER_AREA if the areas with zeros are too many/big they will bring 
        # the values down and the system will automatically react more conservativelly if uncertainty creeps in
        # TODO: find a better way to take into account the holes...
        depth_proj_resized = cv2.resize(depth, (depth.shape[1]//self.depth_decimation_factor,
                                                depth.shape[0]//self.depth_decimation_factor), cv2.INTER_AREA)

        # self.safety_radius and self.proj are in metres
        safety_radius_pixels = int((depth_proj_resized.shape[1]/2)*self.safety_radius/self.proj)
        mask = np.zeros_like(depth_proj_resized)
        mask = cv2.circle(mask, (depth_proj_resized.shape[1]//2,depth_proj_resized.shape[0]//2), safety_radius_pixels, 255, -1)

        depth_proj_resized[mask!=255] = depth_proj_resized.max()
        depth_std = depth_proj_resized[mask==255].std()
        depth_min = depth_proj_resized[mask==255].min()
        
        img_msg = self.cv_bridge.cv2_to_imgmsg((255*depth_proj_resized/max_dist).astype('uint8'), encoding='mono8')
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
        resize_h = 352
        resize_w = int(resize_h*(heatmap.shape[1]/heatmap.shape[0]))
        resize_w = resize_w + 1-(resize_w % 2) # always odd

        heatmap_resized = cv2.resize(heatmap,
                                     (resize_w,resize_h),cv2.INTER_AREA)

        if self.heatmap_filtered is None:
            self.heatmap_filtered = heatmap_resized.astype(float)

        if self.focus_mask_radius is None:
            self.focus_mask_radius = math.sqrt(resize_w**2+resize_h**2)
            self.focus_mask_radius_max = self.focus_mask_radius
            self.safety_radius_pixels = 0
        
        # Add the received heatmap to the buffer
        self.heatmap_filtered += self.beta*(heatmap_resized-self.heatmap_filtered)
        heatmap_resized = self.heatmap_filtered.astype('uint8')

        heatmap_center = np.asarray([heatmap_resized.shape[0]/2, heatmap_resized.shape[1]/2])
        
        # add a black border to avoid problems with distanceTransform
        heatmap_resized[:,0] = 0
        heatmap_resized[:,-1] = 0
        heatmap_resized[0,:] = 0
        heatmap_resized[-1,:] = 0
        
        _,tmp_thresh = cv2.threshold(heatmap_resized,127,255,cv2.THRESH_BINARY)

        radius_mult = 6 if self.landing_status.state == LandingState.AIMING else 2

        if self.landing_status.state == LandingState.AIMING:
            radius_mult = 6
            self.safety_radius_pixels = int(radius_mult*(tmp_thresh.shape[1]/2)*self.safety_radius/self.proj)
        elif self.landing_status.state == LandingState.LANDING:
            radius_mult = 2
            self.safety_radius_pixels = int(radius_mult*(tmp_thresh.shape[1]/2)*self.safety_radius/self.proj)
        elif self.landing_status.state == LandingState.WAITING:
            self.safety_radius_pixels = self.focus_mask_radius
        else:
            self.safety_radius_pixels = self.focus_mask_radius_max
    
        self.focus_mask_radius += (self.safety_radius_pixels - self.focus_mask_radius)*0.1
        mask = np.zeros_like(tmp_thresh)
        mask = cv2.circle(mask, (int(heatmap_center[1]),int(heatmap_center[0])), int(self.focus_mask_radius), 255, -1)
        tmp_thresh[mask!=255] = 0.0

        heatmap_dist_function = cv2.distanceTransform(tmp_thresh, cv2.DIST_L2, maskSize=3)
        cv2.normalize(heatmap_dist_function, heatmap_dist_function, 0, 1.0, cv2.NORM_MINMAX)
               
        # Check area, perimeter, distance from center
        _, dist_thrs = cv2.threshold(heatmap_dist_function, self.dist_func_threshold, 1.0, cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(dist_thrs.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xy_idx = [[1.0,0]] # if nothing is found below, move forward
        objective_values = [0.0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if (area == 0) or (perimeter == 0):
                continue
            mask = np.zeros_like(heatmap_dist_function)
            cv2.drawContours(mask, [cnt], contourIdx=-1, color=(255), thickness=cv2.FILLED)
            # now mask has only the current contour (filled)
            dist_cnt = heatmap_dist_function.copy()
            # using the previously masked heatmap, we expose only the pixels under the contour
            dist_cnt[mask!=255] = 0.0
            cx, cy = np.unravel_index(dist_cnt.argmax(), dist_cnt.shape)
            dist2center = np.sqrt(((heatmap_center-[cx,cy])**2).sum())
            objective = (1/perimeter)*area/(dist2center+1) # complex shapes will have longer perimeter
            objective_values.append(objective)
            x = (heatmap_center[0] - cx)/heatmap_resized.shape[0]
            y = -(heatmap_center[1] - cy)/heatmap_resized.shape[1]
            xy_idx.append([x,y])
        xy_idx = np.asarray(xy_idx)
        desc_order = np.argsort(objective_values)[::-1]
        xy_idx = xy_idx[desc_order]

        yc = int(-(xy_idx[0,0]*heatmap_resized.shape[0]-heatmap_center[0]))
        xc = int(xy_idx[0,1]*heatmap_resized.shape[1]+heatmap_center[1])
        img = cv2.circle(cv2.cvtColor((dist_thrs*255).astype('uint8'), cv2.COLOR_GRAY2BGR), (xc,yc), 10, (0,255,0), -1) # best location (green)
        if xy_idx.shape[0]>1:
            yc = int(-(xy_idx[1,0]*heatmap_resized.shape[0]-heatmap_center[0]))
            xc = int(xy_idx[1,1]*heatmap_resized.shape[1]+heatmap_center[1])
            img = cv2.circle(img, (xc,yc), 10, (255,0,0), -1) # second in line (blue)
        img_msg = self.cv_bridge.cv2_to_imgmsg(img, encoding="bgr8")

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
        curr_t, self.curr_pos, curr_quat = res
        return self.curr_pos[2]
    

    def publish_twist(self, x, y, z):
        twist = Twist()
        # The UAV's maximum bank angle is limited to a very small value
        # and this is why such a simple control works.
        # Additionally, the assumption is that the maximum speed is very low
        # otherwise the moving average used in the semantic segmentation will break.
        # TODO: make it a proper controller ...
        
        self.int_x += x
        self.int_y += y     
        self.int_x = self.int_x if abs(self.int_x) <= self.int_x_sat else np.sign(self.int_x)*self.int_x_sat
        self.int_y = self.int_y if abs(self.int_y) <= self.int_y_sat else np.sign(self.int_y)*self.int_y_sat

        twist.linear.x = float(x * self.gain + self.int_x*self.gain)  # the max bank angle is limited (tiltMax), therefore the gain is here to saturate
        twist.linear.y = -float(y * self.gain + self.int_y*self.gain)
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

        self.get_logger().info(f"Segmentation threshold: {self.landing_status.curr_threshold:0.3f}")
        self.get_logger().info(f"Segmentation success: {self.landing_status.success}")

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

        if self.savefile:
            tmp_dict = {
                'state': str(self.landing_status.state).split('.')[1],
                'is_safe': str(self.landing_status.is_safe),
                'is_clear': str(self.landing_status.is_clear),
                'is_collision_free': str(self.landing_status.is_collision_free),
                'is_flat': str(self.landing_status.is_flat),
                'position': self.curr_pos,
                'conservative_gain': self.landing_status.conservative_gain,
                'loop_freq': 1/self.landing_status.delta_time_sec,
                'curr_threshold': self.landing_status.curr_threshold,
                'success': self.landing_status.success
            }
            self.savedict[int(self.landing_status.elapsed_time_sec*1000)] = tmp_dict


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

            self.proj = math.tan(FOV/2)*self.landing_status.altitude # [m] it's half width because FOV/2

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
            self.req.prompt_engineering = self.prompt_engineering
            self.req.blur_kernel_size = self.blur_kernel_size
            self.req.dynamic_threshold = self.seg_dynamic_threshold

            def future_done_callback(future):
                self.landing_status.curr_threshold = future.result().curr_threshold
                self.landing_status.success = future.result().success
                if future.result().success == True:
                    heatmap_msg = future.result().heatmap
                    xy_err = self.get_xy_error_from_semantics(heatmap_msg)
                    x,y,z = self.state_update(curr_time_sec, xy_err, depth_std, depth_min)
                    self.publish_status()
                    self.publish_twist(x,y,z)
                else:
                    x,y,z = self.state_update(curr_time_sec, np.asarray([[1000.0,1000.0]]), 1000.0, 0.0)
                    self.publish_status()
                    self.publish_twist(x,y,z)
                    self.get_logger().error("Empty heatmap received!")
                self.heatmap_result = None
                if self.landing_status.state == LandingState.LANDED:
                    exit(0)

            self.heatmap_result = self.cli.call_async(self.req)
            self.heatmap_result.add_done_callback(future_done_callback)

            return


    def state_update(self, curr_time_sec, xy_err, depth_std, depth_min):
        estimated_travelled_distance = self.z_speed*self.landing_status.delta_time_sec # TODO:improve this estimation or add some extra margin
        landed_trigger = (self.landing_status.altitude-estimated_travelled_distance) <= self.altitude_landed
        # xy_err are normalised to the center of the image (-1 to 1)
        xy_err = xy_err*self.proj
        xs_err, ys_err = xy_err[0]
        # Very rudimentary filter
        adjusted_err = math.sqrt(xs_err**2+ys_err**2)
        dynamic_threshold = 2*self.safety_radius/self.landing_status.conservative_gain # the segmentation is always noisy, thus the 2x
        self.landing_status.is_clear = adjusted_err < self.safety_radius
        is_clear_dynamic_decision = adjusted_err < dynamic_threshold
        self.get_logger().info(f"Segmentation X,Y ERR, adjusted dist, dynamic threshold: {xs_err:.2f},{-ys_err:.2f},{adjusted_err:.2f},{dynamic_threshold:.2f}")
        # TODO: improve the flat surface definition
        self.landing_status.is_flat = depth_std < self.depth_smoothness
        is_flat_dynamic_decision = depth_std < self.depth_smoothness/self.landing_status.conservative_gain
        # TODO: improve the is_collision_free definition
        self.landing_status.is_collision_free = depth_min > self.safety_radius or landed_trigger
        is_collision_free_dynamic_decision = self.landing_status.is_collision_free

        descend_while_aiming = (self.landing_status.altitude + estimated_travelled_distance) >= self.safe_altitude*1.1
        is_landing = adjusted_err < self.safety_radius and is_collision_free_dynamic_decision and is_flat_dynamic_decision and not descend_while_aiming
        is_aiming = self.landing_status.is_safe and is_clear_dynamic_decision and is_collision_free_dynamic_decision and is_flat_dynamic_decision

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
            if is_landing:
                self.landing_status.state = LandingState.LANDING
            elif is_aiming and self.landing_status.state != LandingState.LANDING:
                self.landing_status.state = LandingState.AIMING
            elif self.landing_status.is_safe and descend_while_aiming:
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
                    if search4new_place_time_passed > self.search4new_place_max_time:
                        # since the system is biased towards places that are close to the UAV, the best way
                        # to avoid a place where landing doesn't work is to move away from it and towards the next candidate
                        self.giveup_landing_timer = 0 # safe place, reset giveup_landing_timer
                        self.search4new_place_timer = 0
                        self.landing_status.state = LandingState.SEARCHING
                else:
                    self.landing_status.state = LandingState.CLIMBING          
            elif is_landing:
                self.landing_status.state = LandingState.LANDING
                self.giveup_landing_timer = 0

        if self.landing_status.state == LandingState.LANDED:
            x = y = z = 0.0
        elif self.landing_status.state == LandingState.SENSOR_ERROR:
            x = y = z = 0.0
        elif self.landing_status.state == LandingState.SEARCHING:
            self.gain = self.input_gain
            x = xs_err
            y = ys_err
            self.int_x = -x  # integrator is only for the AIMING state
            self.int_y = -y  # integrator is only for the AIMING state
            z = 0.0
        elif self.landing_status.state == LandingState.AIMING:
            self.gain = self.input_gain*self.aiming_gain_mult
            x = xs_err
            y = ys_err
            # It's hard to aim when the UAV is too high, so it should descend because 
            # the AIMING state means there's a good landing spot candidate below anyway
            if descend_while_aiming:
                self.z_speed = self.aiming_descending_mult*self.z_gain_landing*self.z_speed_landing*self.landing_status.altitude
                z = -self.z_speed
            else:
                z = 0.0
        elif self.landing_status.state == LandingState.LANDING:
            x = y = 0.0
            self.int_x = self.int_y = 0.0 # integrator is only for the AIMING state
            self.z_speed = self.z_gain_landing*self.z_speed_landing*self.landing_status.altitude
            self.z_speed = (self.z_speed if self.z_speed > self.z_min_speed_landing else self.z_min_speed_landing)
            z = -self.z_speed
        elif self.landing_status.state == LandingState.CLIMBING:
            x = y = 0.0
            z = self.z_speed_climbing
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
    savefile = None
    if len(sys.argv)>1:
        if "debug" in sys.argv[1:]:
            debug = True
        for arg in sys.argv[1:]:
            if "savefile" in arg:
                savefile = arg.split("=")[-1]
    rclpy.init()
    landing_module = LandingModule(debug, savefile)
    try:
        rclpy.spin(landing_module)
    except KeyboardInterrupt:
        pass
    finally:
        if savefile:
            with open(savefile+".json", "w") as file:
                json.dump(landing_module.savedict, file)
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