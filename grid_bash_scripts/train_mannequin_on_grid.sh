cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate
#pip -V #check if env was activated
#cd alias-free-mannequinchallenge
cd mannequinchallenge
python3 train_from_scratch.py --save_weights data_half_with_latent_weight_1e-3_20_epochs_post_weight_init_fix --batch_size 2 --epochs 20 --latent_weight 1e-3 --train_from_scratch
