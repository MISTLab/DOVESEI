import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, Shutdown, SetEnvironmentVariable, RegisterEventHandler, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.event_handlers import (OnExecutionComplete, OnProcessExit,
                                OnProcessIO, OnProcessStart, OnShutdown)



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


def generate_launch_description():

   use_gt_semantics = LaunchConfiguration('use_gt_semantics')
   use_gt_semantics_arg = DeclareLaunchArgument(
      'use_gt_semantics',
      default_value='False'
   )

   max_experiment_time = LaunchConfiguration('max_experiment_time')
   max_experiment_time_arg = DeclareLaunchArgument(
      'max_experiment_time',
      default_value='1200'
   )

   start_delay = LaunchConfiguration('start_delay')
   start_delay_arg = DeclareLaunchArgument(
      'start_delay',
      default_value='10.0'
   )

   debug = LaunchConfiguration('debug')
   debug_arg = DeclareLaunchArgument(
      'debug',
      default_value='False'
   )

   savefile = LaunchConfiguration('savefile')
   savefile_arg = DeclareLaunchArgument(
      'savefile',
      default_value='test_experiment'
   )

   host = LaunchConfiguration('host')
   host_launch_arg = DeclareLaunchArgument(
      'host',
      default_value='carla-container.local'
   )

   town = LaunchConfiguration('town')
   town_launch_arg = DeclareLaunchArgument(
      'town',
      default_value='Town01'
   )

   carla_bridge = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('carla_ros_bridge'), 'carla_ros_bridge.launch.py')]
      ),
      launch_arguments={'town': town, 'host': host}.items(),
      )
   
   standard_semantics = os.path.join(get_package_share_directory('ros2_quad_sim_python'),'cfg/flying_sensor.json')
   gt_standard_semantics = os.path.join(get_package_share_directory('ros2_open_voc_landing_heatmap'),'cfg/semantic_segmentation_added.json')
   objects_definition_file = LaunchConfiguration('objects_definition_file')
   objects_definition_file_arg = DeclareLaunchArgument(
      'objects_definition_file',
      default_value=PythonExpression([f"'{standard_semantics}' if not ", use_gt_semantics, f" else '{gt_standard_semantics}'"])
   )

   carla_spawn_objects = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('carla_spawn_objects'), 'carla_spawn_objects.launch.py')]),
      launch_arguments={'objects_definition_file': objects_definition_file}.items(),
      )
   
   quad_params = {
         'target_frame': ('flying_sensor','Target frame for the flying sensor'), 
         'map_frame': ('map','Map frame for the flying sensor'), 
         'init_pose': ('[100,-100,100,0,0,0]','Initial pose [x,y,z,roll,pitch,yaw]'),
         # Position P gains
         'Px': ('5.0','Position Px gain'),
         'Py': ('5.0','Position Py gain'),
         'Pz': ('2.0','Position Pz gain'),
         # Velocity PID gains
         "Pxdot" : ('2.5','Velocity P x gains'),
         "Dxdot" : ('0.5','Velocity D x gains'),
         "Ixdot" : ('10.0','Velocity I x gains'),
         "Pydot" : ('2.5','Velocity P y gains'),
         "Dydot" : ('0.5','Velocity D y gains'),
         "Iydot" : ('10.0','Velocity I y gains'),
         "Pzdot" : ('4.0','Velocity P z gains'),
         "Dzdot" : ('0.5','Velocity D z gains'),
         "Izdot" : ('5.0','Velocity I z gains'),
         # Attitude P gains
         "Pphi"   : ('4.0','Attitude P phi gain'),
         "Ptheta" : ('4.0','Attitude P theta gain'),
         "Ppsi"   : ('1.5','Attitude P psi gain'),
         # Rate P-D gains
         "Pp" : ('1.5',''),
         "Dp" : ('0.04',''),
         "Pq" : ('1.5',''),
         "Dq" : ('0.04',''),
         "Pr" : ('1.0',''),
         "Dr" : ('0.1',''),
         # Max Velocities (x,y,z) [m/s]
         "uMax" : ('50.0','Max velocity x'),
         "vMax" : ('50.0','Max velocity y'),
         "wMax" : ('50.0','Max velocity z'),
         "saturateVel_separately" : ('True', ''),
         # Max tilt [degrees]
         'tiltMax': ('1.0', 'Maximum angle the UAV is allowed (bank angle)'),
         # Max Rate [rad/s]
         "pMax" : ('100.0',''),
         "qMax" : ('100.0',''),
         "rMax" : ('75.0',''),
         # Minimum velocity for yaw follow to kick in [m/s]
         "minTotalVel_YawFollow" : ('0.1', ''),
         # Include integral gains in linear velocity control
         "useIntegral" : ('True', ''),
         }
   
   quad_params_lcgf = {k: LaunchConfiguration(k) for k in quad_params.keys()}
   quad_params_arg = [DeclareLaunchArgument(k, default_value=v[0], description=v[1]) for k,v in quad_params.items()]
   quad = IncludeLaunchDescription(
      PythonLaunchDescriptionSource([os.path.join(
         get_package_share_directory('ros2_quad_sim_python'), 'quad.launch.py')]),
      launch_arguments=quad_params_lcgf.items(),
      )
   
   lander_params = {
        'img_topic': ('/carla/flying_sensor/rgb_down/image', 'Image topic with a RGB camera pointing downwards'),
        'depth_topic': ('/carla/flying_sensor/depth_down/image', 'Image topic with a downwards depth image'),
        'heatmap_topic': ('/heatmap', 'Topic where to publish the heatmap'),
        'depth_proj_topic': ('/depth_proj', 'Topic where to publish the filtered depth (only the safety radius projection)'),
        'twist_topic': ('/quadctrl/flying_sensor/ctrl_twist_sp', 'Topic to publish the velocity setpoints'),
        'beta': (1/20, 'Gain used with the semantic segmentation filter'),
        'gain': (0.5, 'Gain used with the error to publish velocity setpoints'),
        'aiming_gain_mult': (0.5, 'Multiplier applied to the gain when at AIMING state'),
        'z_speed_landing': (3.0, 'Base value for landing speed'),
        'z_min_speed_landing': (0.5, 'Minimal value for landing speed'),
        'z_gain_landing': (0.02, 'Gain applied to the landing speed according to the altitude'),
        'z_speed_climbing': (6.0, 'Speed used to climb'),
        'depth_smoothness': (0.5, 'Minimal depth smoothness (std)'), # CARLA's values oscillate on flat surfaces
        'depth_decimation_factor': (10, 'Input depth will be resized according to this value'),
        'altitude_landed': (1.5, 'Altitude we can activate the flight controller landing command'),
        'safe_altitude': (50, 'Altitude where there are no obstacles'),
        'safety_radius': (1.5, 'Used to project the UAV to the ground'),
        'safety_threshold': (0.8, 'Used to control the semantic segmentation model'),
        'dist_func_threshold': (0.6, 'Threshold used to binarise the segmentation to find the centre of the free areas'),
        'giveup_after_sec': (5, 'Amount of time the system will wait until it gives up landing at that spot (only below safe_altitude)'),
        'max_depth_sensing': (20, 'Setting telling how far the depth sensor can reach'),
        'use_random_search4new_place': (False, 'When the landing is cancelled it will move towards a random direction (or to the second best place) to avoid getting stuck'),
        'search4new_place_max_time': (60, 'Amount of time moving towards a new direction after a landing was cancelled'),
        'max_landing_time_sec': (300, 'Time used for the conservative gain calculation (exponential decay)'),
        'min_conservative_gain': (0.5, 'Minimum value the conservative gain can reach'),
        'sensor_warm_up_cycles': (5, 'Cycles waiting for the sensors to stabilise'),
        'negative_prompts': (NEGATIVE_PROMPTS, 'Negative prompts, separated by ;'),
        'positive_prompts': (POSITIVE_PROMPTS, 'Positive prompts, separated by ;'),
        'blur_kernel_size': (15, 'Size of the blur kernel'),
        'seg_dynamic_threshold': (0.10, 'Minimum amount of area good for landing as the segmentation will decrease the threshold until it reaches it (0.0 disables it)'),
        'aiming_descending_mult': (0.5, 'Multiplier (in relation to landing speed) for the AIMING state descending speed'),
        'prompt_engineering': (PROMPT_ENGINEERING, 'Prompt modifier where it will add the prompt (negative or positive) where there is {}'),
   }
   lander_params_lcgf = {k: LaunchConfiguration(k) for k in lander_params.keys()}
   lander_params_arg = [DeclareLaunchArgument(k, default_value=str(v[0]), description=v[1]) for k,v in lander_params.items()]
   landing_module= Node(
         package='ros2_open_voc_landing_heatmap',
         executable='lander_publisher',
         name='landing_module',
         # emulate_tty=True,
         on_exit=[LogInfo(msg=PythonExpression(["str('>'*15 + ","' savefile='+'",savefile,".json' if len('", savefile, "') else '')"])), Shutdown()],
         arguments=[PythonExpression(["'debug' if ", debug, " else ''"]), PythonExpression(["'savefile='+'",savefile,"' if len('", savefile, "') else ''"])],
         parameters=[{i[0]:i[1][0]} for i in lander_params.items()]
      )

   generate_landing_heatmap = Node(
         package='ros2_open_voc_landing_heatmap',
         executable=PythonExpression(["'getlandingheatmap_service' if not ", use_gt_semantics, " else 'getlandingheatmap_gt_service'"]),
         name='generate_landing_heatmap',
         # emulate_tty=True,
         on_exit=Shutdown(),
         parameters=[{'img_topic':lander_params['img_topic'][0]}]
      )

   return LaunchDescription([
      SetEnvironmentVariable(name='RCUTILS_COLORIZED_OUTPUT', value='1'),
      use_gt_semantics_arg,
      max_experiment_time_arg,
      start_delay_arg,
      debug_arg,
      savefile_arg,
      host_launch_arg,
      town_launch_arg,
      carla_bridge,
      objects_definition_file_arg,
      carla_spawn_objects,
      *quad_params_arg,
      quad,
      *lander_params_arg,
      TimerAction(
            period=start_delay,
            actions=[LogInfo(msg='Starting the self-landing nodes...'), generate_landing_heatmap, landing_module]
      ),
      TimerAction(
            period=max_experiment_time,
            actions=[LogInfo(msg='MAX_EXPERIMENT_TIME REACHED!!!!!'), Shutdown()]
      )
   ])