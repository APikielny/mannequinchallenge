cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate
#pip -V #check if env was activated
cd alias-free-mannequinchallenge
#cd mannequinchallenge
python3 train_from_scratch.py --save_weights 'data_half_anti_alias_upsample_and_downsample_radial_pad_crit_sampling_20_epochs_post_weight_init_fix' --batch_size 4 --epochs 20 --plot_losses --train_from_scratch
#python3 train_from_scratch.py --save_weights 'upsample_arch_dummy_testing' --batch_size 2 --epochs 20 --plot_losses #--train_from_scratch
