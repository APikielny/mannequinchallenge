model_name=${1:-"data_half_anti_alias_upsample_and_downsample_radial_pad_non_crit_sampling_20_epochs_post_weight_init_fix"}

cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate
#pip -V #check if env was activated
cd alias-free-mannequinchallenge #TODO before changing this make sure the code has moved over to each repo
#python3 test_davis_videos.py --weights ${1}
python3 test_davis_videos.py --weights ${model_name} --visualize

#bash L2_frame_comparison ${model_name}
