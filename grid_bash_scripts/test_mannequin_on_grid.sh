model_name=${1:-"0_epochs_trying_to_fix_weight_init_constant_0_weights"}

cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate
#pip -V #check if env was activated
cd mannequinchallenge
python3 test_davis_videos.py --weights ${model_name}

#bash L2_frame_comparison ${model_name}
