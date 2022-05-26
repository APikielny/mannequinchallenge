# ###################
# ### convert to video ####
# ###################
model_name=${1:-"data_half_with_latent_weight_1e-3_20_epochs_post_weight_init_fix"}
data=${3:-"adam_translate"}
epoch=${2:-"20"}

echo "writing depth frames to video"
mkdir -p test_data/videos/${data}
ffmpeg -framerate 60 -f image2 -i test_data/viz_predictions/${data}/${model_name}/epoch_${epoch}/frame%0d.jpg -vcodec mpeg4 -c:v libx264 -y test_data/videos/${data}/${model_name}.mp4 2> /dev/null
FILE=test_data/videos/${data}/${model_name}.mp4
if [ -f "$FILE" ]; then
    echo "Successfully wrote depth video to $FILE."
else
    echo "Something went wrong when writing to $FILE. Exiting."
    exit
fi
