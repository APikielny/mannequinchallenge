# ###################
# ### convert to video ####
# ###################
model_name=${1:-"data_half_with_latent_weight_1e-5_20_epochs_post_weight_init_fix"}
data=${2:-"brown-campus-data"}
id=${3:-"arch"}
# epoch=${3:-"20"}

echo "writing depth frames to video"
mkdir -p test_data/videos/${data}/${id}
# ffmpeg -framerate 60 -f image2 -i test_data/viz_predictions/generating_result_videos/${data}/${id}/images/${model_name}/frame%04d.jpg -vcodec mpeg4 -c:v libx264 -y test_data/videos/${data}/${id}/${model_name}.mp4 #2> /dev/null

ffmpeg -framerate 30 -f image2 -i test_data/viz_predictions/generating_result_videos/${data}/${id}/images/${model_name}/frame%04d.jpg -y -vcodec libx264 -crf 25  -pix_fmt yuv420p test_data/videos/${data}/${id}/${model_name}.mp4 2> /dev/null
#add this to write text on the video: -vf "drawtext=fontfile=/path/to/font.ttf:text='${id}':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=w-tw-5:y=h-th-5"

FILE=test_data/videos/${data}/${id}/${model_name}.mp4
if [ -f "$FILE" ]; then
    echo "Successfully wrote depth video to $FILE."
else
    echo "Something went wrong when writing to $FILE. Exiting."
    exit
fi
