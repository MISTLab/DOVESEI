#!/bin/bash

experiment_folders=($(ls -d experiment_*)) # Create the experiment name

for i in "${!experiment_folders[@]}"
do
  echo "Processing folder ${experiment_folders[i]}"
  cd ${experiment_folders[i]}
  ls rgb* | grep png | xargs cat | ffmpeg -f image2pipe -i - ${experiment_folders[i]}.mp4
  cd ..
done
