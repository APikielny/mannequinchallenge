#!/bin/bash
###########
# This script will train a model, then run inference and check consistency metrics on the ouput # 
##########

set -e #exit if any cmd fails

###################
### Params ########
###################
lr=${1:-"0.001"}
model_prefix=${2:-"new_model"} #where to save weights
model_data_train=${3:-"translate"} #what data to use for training
model_data_test=${4:-${model_data_train}} #default to same as train data
model_name=${model_prefix}_data_${model_data_train}


echo "learning rate: " ${lr}
echo "model name: " ${model_name}
echo "model train data: " ${model_data_train}
echo "model test data: " ${model_data_test}

###################
### Train model ###
###################
echo "training model..."
python train_davis_videos.py --save_weights ${model_name} --marc_data_train ${model_data_train} > /dev/null #the >/dev/null suppresses output from the python script
#check if model was saved

FILE=checkpoints/test_local/${model_name}_net_G.pth
if [ -f "$FILE" ]; then
    echo "Successfuly wrote model to $FILE."
else 
    echo "Something went wrong when training or writing to $FILE. Exiting."
    exit
fi

###################
### Test model ####
###################
echo "running inference (output frames)..."
python test_davis_videos.py --weights ${model_name} --marc_data_inference ${model_data_test} > /dev/null
echo "frames written out to test_data/viz_predictions/${model_data_test}/${model_name}"

#TODO convert to ffmpeg here

###################
### Metrics #######
###################
echo "calculating consistency metrics..."
python metrics.py --folder test_data/viz_predictions/${model_data_test}/${model_name} > /dev/null
echo "metrics written out to L2_frame_comparisons/${model_data_test}/${model_name}_L2_plot.png"