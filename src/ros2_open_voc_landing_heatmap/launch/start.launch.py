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
                    "tree, bush, tall-plant", 
                    "water, lake, river, swimming pool",
                    "people, crowd", 
                    "vehicle, car, train", 
                    "lamp-post, transmission-line", 
                    "fence, wall, hedge", 
                    "road, street, avenue, highway, drive, lane",
                    "stairs, steps, footsteps"]
positive_prompts = ["grass, backyard, frontyard, courtyard, lawn", 
                    "sports-field, park, open-area, open-space"] 

PROMPT_ENGINEERING = "a bird's eye view of a {}, ingame screen shot, bad graphics"

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
         'init_pose': ('[100,-100,100,0,0,0]','Initial pose'),
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
         'tiltMax': ('3.0', ''),
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
   
   generate_landing_heatmap = Node(
         package='ros2_open_voc_landing_heatmap',
         executable=PythonExpression(["'getlandingheatmap_service' if not ", use_gt_semantics, " else 'getlandingheatmap_gt_service'"]),
         name='generate_landing_heatmap',
         # emulate_tty=True,
         on_exit=Shutdown(),
      )
   
   lander_params = {
        'img_topic': ('/carla/flying_sensor/rgb_down/image', ''),
        'depth_topic': ('/carla/flying_sensor/depth_down/image', ''),
        'heatmap_topic': ('/heatmap', ''),
        'depth_proj_topic': ('/depth_proj', ''),
        'twist_topic': ('/quadctrl/flying_sensor/ctrl_twist_sp', ''),
        'beta': (1/20, ''),
        'gain': (0.5, ''),
        'aiming_gain_mult': (0.5, ''),
        'z_speed_landing': (3.0, ''),
        'z_speed_climbing': (6.0, ''),
        'depth_smoothness': (0.5, ''), # CARLA's values oscillate on flat surfaces
        'depth_decimation_factor': (10, ''),
        'altitude_landed': (1, ''),
        'safe_altitude': (50, ''),
        'safety_radius': (1.5, ''),
        'safety_threshold': (0.8, ''),
        'dist_func_threshold': (0.6, ''),
        'giveup_after_sec': (5, ''),
        'max_depth_sensing': (20, ''),
        'use_random_search4new_place': (False, ''),
        'search4new_place_max_time': (60, ''),
        'max_seg_height': (17, ''),
        'max_landing_time_sec': (300, ''),
        'min_conservative_gain': (0.5, ''),
        'sensor_warm_up_cycles': (5, ''),
        'negative_prompts': (NEGATIVE_PROMPTS, ''),
        'positive_prompts': (POSITIVE_PROMPTS, ''),
        'blur_kernel_size': (15, ''),
        'seg_dynamic_threshold': (0.10, ''),
        'prompt_engineering': (PROMPT_ENGINEERING, ''),
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