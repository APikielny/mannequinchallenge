#!/bin/bash

lr=${1:-"0.001"}
model_name=${2:-"new_model"}

echo "learning rate: " ${lr}
echo "model name: " ${model_name}

echo "training model"
python train_davis_videos.py --save_weights new_model

echo "running inference (output frames)"
python test_davis_videos.py --weights new_model

#TODO convert to ffmpeg here

echo "calculating consistency metrics"
python metrics.py --folder test_data/viz_predictions/breakdance-flare/breakdance-flare_original_model