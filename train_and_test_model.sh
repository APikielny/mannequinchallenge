#!/bin/bash
###########
# This script will train a model, then run inference and check consistency metrics on the ouput 
##########

set -e #exit if any command fails

###################
### Params ########
###################
lr=${1:-"0.01"}
model_prefix=${2:-"latent_constrained"} #where to save weights
model_data_train=${3:-"static"} #what data to use for training
model_data_test=${4:-${model_data_train}} #default to same as train data
model_name=${model_prefix}_lr_${lr}_data_${model_data_train}

echo "
###############
"
echo "learning rate: " ${lr}
echo "model train data: " ${model_data_train}
echo "model test data: " ${model_data_test}
echo "model name: " ${model_name}
echo "
###############
"



###################
### Train model ###
###################

#TODO if this model already exists, maybe skip?

echo "training model..."
python train_davis_videos.py --save_weights ${model_name} --marc_data_train ${model_data_train} --lr ${lr} > /dev/null #the >/dev/null suppresses output from the python script
#check if model was saved

FILE=checkpoints/test_local/${model_name}_net_G.pth
if [ -f "$FILE" ]; then
    echo "Successfully wrote weights to $FILE."
else 
    echo "Something went wrong when training or writing to $FILE. Exiting."
    exit
fi
echo "
###############
"


###################
### Test model ####
###################
echo "running inference..."
python test_davis_videos.py --weights ${model_name} --marc_data_inference ${model_data_test} > /dev/null
echo "frames written out to test_data/viz_predictions/${model_data_test}/${model_name}"

echo "
###############
"


###################
### convert to video ####
###################
echo "writing frames to video"
mkdir -p test_data/videos/${model_data_test}
ffmpeg -framerate 60 -f image2 -i test_data/viz_predictions/${model_data_test}/${model_name}/frame%04d.jpg -vcodec mpeg4 -c:v libx264 -y test_data/videos/${model_data_test}/${model_name}.mp4 2> /dev/null
FILE=test_data/videos/${model_data_test}/${model_name}.mp4
if [ -f "$FILE" ]; then
    echo "Successfully wrote video to $FILE."
else 
    echo "Something went wrong when writing to $FILE. Exiting."
    exit
fi

echo "
###############
"

###################
### Metrics #######
###################
echo "calculating consistency metrics..."
mkdir -p L2_frame_comparisons/${model_data_test}
python metrics.py --L2_folder test_data/viz_predictions/${model_data_test}/${model_name} > /dev/null
echo "metrics written out to L2_frame_comparisons/${model_data_test}/${model_name}_L2_plot.png"