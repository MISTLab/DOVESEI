# ros2_open_voc_landing_heatmap

Work-in-progress... the repo is public because it's easier to access from shared computers this way ;)

Follow the instructions from https://github.com/ricardodeazambuja/ros2_quad_sim_python, but using:
```
launch_ros2_desktop.sh -g --image ricardodeazambuja/ros2_quad_sim_python:pytorch
```
(You can find the Dockerfiles [here](https://github.com/ricardodeazambuja/ros2_quad_sim_python/tree/main/docker))

It's necessary to limit the maximum bank angle (tiltMax) or the UAV will oscillate at a high altitude making it very 
hard for the system to work as it expects the downward facing camera to be approximately perpendicular to the ground.
```
ros2 launch ros2_quad_sim_python launch_everything.launch.py init_pose:=[100,-100,100,0,0,0] tiltMax:=1.0
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
Important parameters (append `--ros-args -p <parameter_name>:=<parameter_value>`):
* `model_calib_cte`: gain applied to the output of the softmax to calibrate the model (same value used for all prompts)
* `blur_kernel_size`: size of a square kernel used to smooth the pattern left by the ViT patches
* `safety_threshold`: threshold used on the fused logits to generate the places for landing mask (before the distance gradient)


Launch the node that will publish the twist messages:
```
ros2 run ros2_open_voc_landing_heatmap lander_publisher --ros-args -p mov_avg_size:=10
```
Important parameters (append `--ros-args -p <parameter_name>:=<parameter_value>`):
* `mov_avg_size`: number of heatmaps averaged before calculating the direction to move
* `z_speed`: speed setpoint [m/s] that the UAV will move in the z direction
* `depth_smoothness`: value used to control the smoothness of the surface 
* `altitude_landed`: altitude considered close enough to the ground to use the flight controller landing procedure
* `safe_altitude`: altitude above which there are no obstacles
* `safety_radius`: minimum clearance for the UAV
* `giveup_after_sec`: amount of time before the UAV gives up the current landing spot after continuosly detecting problems
* `use_random_search4new_place`: selects between a random new direction or calculate using the current heatmap
* `search4new_place_max_time`: amount of time moving towards a new direction after a failed landing tentative
* `heatmap_mask_erosion`: amount of pixels eroded on the heatmap mask to fight semantic segmentation noise

It's also possible to generate the heatmap using CARLA's semantic segmentation sensor (must be enabled in the [json config file](https://github.com/ricardodeazambuja/ros2_quad_sim_python/blob/24747bb8c7d0cb3f35087b4154da1cfbec49527a/src/ros2_quad_sim_python/cfg/flying_sensor_full.json)) as a way to compare against the open vocabulary one without changing the `lander_publisher` node:
```
ros2 run ros2_open_voc_landing_heatmap getlandingheatmap_gt_service
```


## TODO
* Improve the code
* Fix the service call_async / future / MultiThreadedExecutor mess
* Write launch files