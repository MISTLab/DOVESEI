# ros2_open_voc_landing_heatmap

Follow the instructions from https://github.com/ricardodeazambuja/ros2_quad_sim_python, but using:
```
launch_ros2_desktop.sh -g --image ricardodeazambuja/ros2_quad_sim_python:pytorch
```
(You can find the Dockerfiles [here](https://github.com/ricardodeazambuja/ros2_quad_sim_python/tree/main/docker))
```
ros2 launch ros2_quad_sim_python launch_everything.launch.py init_pose:=[100,-100,100,0,0,0] tiltMax:=0.5
```

And after cloning this repo in your workspace:
```
colcon build --symlink-install --packages-select ros2_open_voc_landing_heatmap ros2_open_voc_landing_heatmap_srv
```

```
. install/setup.bash
```

Start the node with the heatmap generation from open vocabulary service:
```
ros2 run ros2_open_voc_landing_heatmap getlandingheatmap_service
```

Launch the node that will publish the twist messages:
```
ros2 run ros2_open_voc_landing_heatmap lander_publisher --ros-args -p mov_avg_size:=10
```

## TODO
* Improve the code
* Fix the service call_async / future / MultiThreadedExecutor mess
* Write launch files