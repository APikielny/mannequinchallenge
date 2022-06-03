# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch import float16
from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int,
                                 default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument(
            '--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument(
            '--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100,
                                 help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_decay_epoch', type=int, default=8,
                                 help='# of epoch to linearly decay learning rate to zero')
        self.parser.add_argument('--lr_policy', type=str, default='step',
                                 help='learning rate policy: lambda|step|plateau')

        self.parser.add_argument(
            '--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument(
            '--lr', type=float, default=0.0004, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument(
            '--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument(
            '--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--pool_size', type=int, default=50,
                                 help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data argumentation')

        # added by Adam
        self.parser.add_argument('--visualize', action='store_true',
                                 help='visualize latent space')

        self.parser.add_argument('--weights', type=str,
                                 help='to load in new custom models, instead of the default models')

        self.parser.add_argument('--save_weights', type=str,
                                 help='to save new custom models')

        self.parser.add_argument('--marc_data_train', type=str,
                                 help='which videos to use for training, options rotate, translate, static')

        self.parser.add_argument('--marc_data_inference', type=str,
                                 help='which videos to use for testing, options rotate, translate, static')
        
        self.parser.add_argument('--batch_size', type=int, default = 4,
                                 help='batch size')
        
        self.parser.add_argument('--epochs', type=int, default = 2,
                                 help='number of epochs to train for')

        self.parser.add_argument('--plot_losses', action='store_true',
                                 help='plot latent and supervision losses for each epoch')
        self.parser.add_argument('--latent_weight', type=float,
                                help='weight to use when combining latent and supervision losses')
        self.parser.add_argument('--train_from_scratch', action='store_true',
                                help='if training from scratch, do not load an old model weights')
                                

        self.parser.add_argument('--use_1x1_conv', action="store_true",
                                help="use 1x1 convolutions to achieve rotational equivariance. If passing this flag, MUST also pass a --scale to enable Fourier features")
        self.parser.add_argument('--scale', type=float,
                                help="scale/sampling frequency to use for Gaussian Fourier Features. If not specified, don't use Fourier Features.")
        self.parser.add_argument('--anti_alias_upsample', action="store_true",
                                help="use windowed sinc lowpass filter for upsampling")
        self.parser.add_argument('--anti_alias_downsample', action="store_true",
                                help="use windowed sinc lowpass filter for downsampling")

        #for accuracy metric
        self.parser.add_argument('--accuracy_test_list', type=str,
                                help='a test list to use for inference')
        self.parser.add_argument('--accuracy_dataset', type=str,
                                help='the dataset being used')
        self.parser.add_argument('--viz_folder', type=str,
                                help='folder in viz predictions for saving data')
        self.parser.add_argument('--accuracy_id', type=str,
                                help='the id being used')

        self.isTrain = True
