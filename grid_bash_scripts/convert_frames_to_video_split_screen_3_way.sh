# ###################
# ### convert to video ####
# ###################
num_inputs=${1:-"2"}
data=${2:-"brown-campus-data"}
id=${3:-"mg2"}
# epoch=${3:-"20"}

model_name_1=${4:-"data_half_with_latent_weight_0_20_epochs_post_weight_init_fix"}
text_1=${5:-"Baseline"}
model_name_2=${6:-"data_half_with_latent_weight_1e-5_20_epochs_post_weight_init_fix"}
text_2=${7:-"1e-5"}
model_name_3=${6:-"10_MORE_EPOCHS_FINE_TUNE_data_half_with_latent_weight_1e-1_10_epochs_post_weight_init_fix"}
text_3=${7:-"1e-1"}

/data/jhtlab/apikieln/mannequinchallenge
echo "writing depth frames to video"
mkdir -p test_data/videos/${data}/${id}
ffmpeg -framerate 30 -f image2 \
        -i test_data/viz_predictions/generating_result_videos/${data}/${id}/images/${model_name_1}/frame%04d.jpg \
        -i test_data/viz_predictions/generating_result_videos/${data}/${id}/images/${model_name_2}/frame%04d.jpg \
        -i test_data/viz_predictions/generating_result_videos/${data}/${id}/images/${model_name_3}/frame%04d.jpg \
        -y -vcodec libx264 -crf 25  -pix_fmt yuv420p \
        -filter_complex \
"[0:v]crop=iw/2:ih:iw/2:0,drawtext=fontfile=/usr/share/fonts/TTF/Vera.ttf:fontsize=45:text=${text_1}:x=(w-text_w)-10:y=h-text_h-10,setpts=PTS-STARTPTS[l]; \
[1:v]crop=iw/2:ih:iw/2:0,drawtext=fontfile=/usr/share/fonts/TTF/Vera.ttf:fontsize=45:text=${text_2}:x=(w-text_w)-10:y=h-text_h-10,setpts=PTS-STARTPTS[r]; \
[2:v]crop=iw/2:ih:iw/2:0,drawtext=fontfile=/usr/share/fonts/TTF/Vera.ttf:fontsize=45:text=${text_3}:x=(w-text_w)-10:y=h-text_h-10,setpts=PTS-STARTPTS[r2]; \
[l][r][r2]hstack=inputs=3[out]" \
    -map "[out]" \test_data/videos/${data}/${id}/3_way_split_${id}.mp4 #2> /dev/null

#add this to write text on the video: -vf "drawtext=fontfile=/path/to/font.ttf:text='${id}':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=w-tw-5:y=h-th-5"

FILE=test_data/videos/${data}/${id}/${model_name_1}_and_${model_name_2}.mp4
if [ -f "$FILE" ]; then
    echo "Successfully wrote depth video to $FILE."
else
    echo "Something went wrong when writing to $FILE. Exiting."
    exit
fi
