cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate
#pip -V #check if env was activated
cd mannequinchallenge-alias-fourier-merge
python3 train_from_scratch.py --save_weights merge_branches_dummy_model_testing_with_fourier_if_statement --batch_size 2 --epochs 2 --train_from_scratch --scale 0.1 --anti_alias_downsample
