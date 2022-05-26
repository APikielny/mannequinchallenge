# ###################
# ### convert to video ####
# ###################
num_inputs=${1:-"2"}
data=${2:-"data-test-small"}
id=${3:-"ed4ec0767a7b474d"}
viz_dir=${4:-"accuracy_testing"}
# epoch=${3:-"20"}

model_name_1=${4:-"data_half_with_latent_weight_0_20_epochs_post_weight_init_fix"}
text_1=${5:-"Baseline"}
model_name_2=${6:-"data_half_anti_alias_downsample_only_radial_pad_non_crit_sampling_20_epochs_post_weight_init_fix"}
text_2=${7:-"Anti-aliased Downsampling"}

cd /data/jhtlab/apikieln/mannequinchallenge

echo "writing depth frames to video"
mkdir -p test_data/videos/${data}/${id}
ffmpeg -framerate 10 -f image2 \
        -i test_data/viz_predictions/${viz_dir}/${data}/${id}/images/${model_name_1}/frame%04d.jpg \
        -framerate 10 -f image2 \
        -i test_data/viz_predictions/${viz_dir}/${data}/${id}/images/${model_name_1}/frame%04d.jpg \
        -framerate 10 -f image2 \
        -i ../alias-free-mannequinchallenge/test_data/viz_predictions/${viz_dir}/${data}/${id}/images/${model_name_2}/frame%04d.jpg \
        -y -vcodec libx264 -crf 25  -pix_fmt yuv420p \
        -filter_complex \
"[0:v]crop=iw/2:ih:0:0,drawtext=fontfile=/usr/share/fonts/TTF/Vera.ttf:fontsize=25:text=${text_1}:x=(w-text_w)-10:y=h-text_h-10,setpts=PTS-STARTPTS[l1];
[1:v]crop=iw/2:ih:iw/2:0,drawtext=fontfile=/usr/share/fonts/TTF/Vera.ttf:fontsize=25:text=${text_1}:x=(w-text_w)-10:y=h-text_h-10,setpts=PTS-STARTPTS[l]; \
[2:v]crop=iw/2:ih:iw/2:0,drawtext=fontfile=/usr/share/fonts/TTF/Vera.ttf:fontsize=25:text=${text_2}:x=(w-text_w)-10:y=h-text_h-10,setpts=PTS-STARTPTS[r]; \
[l1][l][r]hstack=inputs=3[out]" \
    -map "[out]" \test_data/videos/${data}/${id}/downsample_and_baseline_id_${id}_with_input.mp4 #2> /dev/null

#add this to write text on the video: -vf "drawtext=fontfile=/path/to/font.ttf:text='${id}':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:x=w-tw-5:y=h-th-5"

FILE=test_data/videos/${data}/${id}/${model_name_1}_and_${model_name_2}.mp4
if [ -f "$FILE" ]; then
    echo "Successfully wrote depth video to $FILE."
else
    echo "Something went wrong when writing to $FILE. Exiting."
    exit
fi
