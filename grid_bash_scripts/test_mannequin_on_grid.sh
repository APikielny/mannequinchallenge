model_name=${1:-"best_depth_Ours_Bilinear_inc_3"} #this is google's model.

cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate
#pip -V #check if env was activated
cd mannequinchallenge
python3 test_davis_videos.py --weights ${model_name}
