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


Launch the node that will publish the twist messages:
```
ros2 run ros2_open_voc_landing_heatmap lander_publisher --ros-args -p mov_avg_size:=10
```
Important parameters (append `--ros-args -p <parameter_name>:=<parameter_value>`):
* `safety_threshold`: threshold used on the fused logits to generate the places for landing mask (before the distance gradient)
* `mov_avg_size`: number of heatmaps averaged before calculating the direction to move
* 'gain': gain used on the XY error to define the velocity to move (usually the values get satured by the maximum bank angle, `tiltMax`)
* `z_speed`: speed setpoint [m/s] that the UAV will move in the z direction
* `depth_smoothness`: value used to control the smoothness of the surface 
* `altitude_landed`: altitude considered close enough to the ground to use the flight controller landing procedure
* `safe_altitude`: altitude above which there are no obstacles
* `safety_radius`: minimum clearance for the UAV
* `giveup_after_sec`: amount of time before the UAV gives up the current landing spot after continuosly detecting problems
* `use_random_search4new_place`: selects between a random new direction or calculate using the current heatmap
* `search4new_place_max_time`: amount of time moving towards a new direction after a failed landing tentative
* `heatmap_mask_erosion`: amount of pixels eroded on the heatmap mask to fight semantic segmentation noise
* `max_landing_time_sec`: maximum time expected for the landing (controls how the `safety_threshold` and flatness are reduced as the elapsed time increases)

It's also possible to generate the heatmap using CARLA's semantic segmentation sensor (must be enabled in the [json config file](https://github.com/ricardodeazambuja/ros2_quad_sim_python/blob/24747bb8c7d0cb3f35087b4154da1cfbec49527a/src/ros2_quad_sim_python/cfg/flying_sensor_full.json)) as a way to compare against the open vocabulary one without changing the `lander_publisher` node:
```
ros2 run ros2_open_voc_landing_heatmap getlandingheatmap_gt_service
```

## How it works internally:
The system has two distinct phases:
1. Search for a landing spot from a safe altitude
2. Descending while checking for changes or obstacles

### 1. Search for a landing spot from a safe altitude
An important assumption at this phase is that the UAV can move in the XY plane freely as there are no obstacles at this altitude in this particular area (e.g. 100m). Thus the system will only use its downfacing RGB sensor to search for a place to land.     
The captured RGB image is passed through a semantic segmentation model to detect classes that the UAV must avoid. Currently this is done using CLIPSeg and the following prompts: `["building", "tree", "road", "water", "transmission lines", "lamp post", "vehicle", "people"]`. After a softmax layer, the outputs of the model are calibrated (`model_calib_cte`), fused together (maximum values), and inverted to bring free pixels values towards one. Since CLIPSeg is based on visual transformers (ViT), it divides the input image into patches and this causes artifacts to appear in the output. Therefore the fused outputs are smoothed (blur, `blur_kernel_size`). With this smoother result, a mask is generated (`safety_threshold`) and eroded (`heatmap_mask_erosion`) to remove small noisy regions. One of the goals is try to land as close as the current position, therefore a distance gradient is applied on top of the eroded mask giving higher values to free pixels closer to the centre of the image. Finally, the resolution is reduced to fight noise and use less CPU in the next steps:     
![image](https://github.com/ricardodeazambuja/ros2_open_voc_landing_heatmap/assets/6606382/8da0e4af-cb85-4a4a-87e6-555c64ed8552)

Having the (low resolution) heatmap ready, the system averages a certain number of frames (`mov_avg_size`) before it searches for the brightest pixel. The position of the brightest pixel will define the direction commanded to the UAV to move. The system will stay in this phase until the brightest pixel, therefore the XY error, is below a certain threshold and at that moment it will start descending (`z_speed`) until it breaches the safe altitude (`safe_altitude`) when it switches to the next phase.
![image](https://github.com/ricardodeazambuja/ros2_open_voc_landing_heatmap/assets/6606382/3d6fe182-318c-4428-ad33-cd4031a4f30b)

### 2. Descending while checking for changes or obstacles
The descending phase is triggered by the current altitude (`safe_altitude`). At this situation the movements in the XY plane are not allowed anymore and the system checkes for consistent collisions, flatness, or changes in the heatmap. The collisions and flatness check uses the received depth image, but only the region corresponding to the UAV's projection on the ground. To avoid noise, the system only gives up a landing spot after problems are detected consistently for a certain amount of time (`giveup_after_sec`). If the problems persist after `giveup_after_sec`, the UAV will climb again to a safe altitude and search for a new place. When it is searching for a new place, it can use the current heatmap or a random direction (`use_random_search4new_place`), and move towards that direction for the amount of time specified (`search4new_place_max_time`).
![image](https://github.com/ricardodeazambuja/ros2_open_voc_landing_heatmap/assets/6606382/1df3e9f6-9e8a-4777-aa67-fe7f8f113079)
A conservative gain is calculated based on the elapsed time since the start of the procedure and the 
The system will activate the flight controller's internal landing procedure when it reaches a certain altitude (`altitude_landed`).


## Parameters that can make the system more or less conservative:
* `zero_error_eps`: used as the threshold for the xy error calculated using the heatmap. Increasing this value will make the system less conservative
* `max_sem_height`: maximum height resolution of the heatmap. Decreasing this value will fuse more pixels into one, reducing noise
* `giveup_after_sec`: maximum time of continuous non-zero xy error, flatness and obstacle detection. Increasing this value will make the system less conservative
* `safety_radius`: radius [m] of the UAV projection on the ground. Decreasing this value will leave more room for errors making the system less conservative
* `mov_avg_size`: number of heatmap frames averaged. Increasing this value will reduce disruptions by noisy readings
* `heatmap_mask_erosion`: pixels used to erode the mask generated from the segmentation fused logits. Decreasing this value will make the system less conservative as the high valued areas (places good to land) will expand
* `depth_smoothness`: depth value [m] for flatness and collision estimations. Increasing this value will make the system less conservative
* `z_speed`: z speed setpoint. Increasing this value may lead to a less conservative system (it will play a role with the mov_avg_size, though)
* `max_landing_time_sec`: maximum time expected before landing. Decreasing this value will make the `safety_threshold` and minimum flatness acceptable values decrease faster

## TODO
* Improve code structure
* Define better flatness and obstacle detection algorithms (while still keeping the CPU usage low)
* Fix the service call_async / future / MultiThreadedExecutor mess
* Write launch files
