#!/bin/bash
docker build --platform linux/amd64 -t ros2_open_voc_landing_heatmap:$USER -f Dockerfile.localuser --build-arg UID=$(id -u) --build-arg GID=$(id -g) .