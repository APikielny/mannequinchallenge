model_name=${1:-"data_half_with_anti_alias_upsample"}

cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate
#pip -V #check if env was activated
cd alias-free-mannequinchallenge
#python3 test_davis_videos.py --weights ${1}
python3 test_davis_videos.py --weights ${model_name}

#bash L2_frame_comparison ${model_name}
