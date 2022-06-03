cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate
#pip -V #check if env was activated
cd alias-free-mannequinchallenge
#cd mannequinchallenge
python3 train_from_scratch.py --save_weights 'data_half_FINE_TUNE_baseline_no_alias_free_10_epochs_post_weight_init_fix' --batch_size 4 --epochs 10 --plot_losses
#python3 train_from_scratch.py --save_weights 'upsample_arch_dummy_testing' --batch_size 2 --epochs 20 --plot_losses #--train_from_scratch
