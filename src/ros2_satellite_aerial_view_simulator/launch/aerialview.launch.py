import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, Shutdown, SetEnvironmentVariable, RegisterEventHandler, TimerAction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.event_handlers import (OnExecutionComplete, OnProcessExit,
                                OnProcessIO, OnProcessStart, OnShutdown)



def generate_launch_description():

   lat = LaunchConfiguration('lat')
   lat_arg = DeclareLaunchArgument(
      'lat',
      default_value='48.858327718853104'
   )

   lon = LaunchConfiguration('lon')
   lon_arg = DeclareLaunchArgument(
      'lon',
      default_value='2.294309636169546'
   )

   z_init = LaunchConfiguration('z_init')
   z_init_arg = DeclareLaunchArgument(
      'z_init',
      default_value='100.0'
   )

   aerialimages = Node(
         package='ros2_satellite_aerial_view_simulator',
         executable='aerialimages_publisher',
         name='aerialimages_module',
         # emulate_tty=True,
         parameters=[{'lat':lat}, {'lon':lon}, {'z_init':z_init}]
      )

   return LaunchDescription([
      SetEnvironmentVariable(name='RCUTILS_COLORIZED_OUTPUT', value='1'),
      lat_arg,
      lon_arg,
      z_init_arg,
      aerialimages
   ])