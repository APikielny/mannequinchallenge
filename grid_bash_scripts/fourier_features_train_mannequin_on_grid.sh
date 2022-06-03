cd /data/jhtlab/apikieln
source ./venv-mannequin/bin/activate
#pip -V #check if env was activated
cd fourier-features-mannequin
#cd mannequinchallenge
#python3 train_from_scratch.py --save_weights 'many_epochs_dummy_test' --batch_size 4 --epochs 100 --plot_losses --train_from_scratch --lr 0.001 --scale 1
python3 train_from_scratch.py --save_weights 'data_half_fourier_features_scale_0.001_lr_0.001_20_epochs_post_weight_init_fix' --batch_size 4 --epochs 20 --plot_losses --train_from_scratch --lr 0.001 --scale 0.001

#1x1 conv:
#python3 train_from_scratch.py --save_weights 'data_half_1x1_conv_fourier_features_scale_2_lr_0.001_anti_alias_up_and_down_radial_pad_non_crit_sampling_30_epochs_post_weight_init_fix' --batch_size 1 --epochs 30 --plot_losses --train_from_scratch --lr 0.001 --use_1x1_conv
#python3 train_from_scratch.py --save_weights 'upsample_arch_dummy_testing' --batch_size 2 --epochs 20 --plot_losses #--train_from_scratch
