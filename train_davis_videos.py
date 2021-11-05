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

import time
from os import system
import torch
from options.train_options import TrainOptions
from loaders import aligned_data_loader
from models import pix2pix_model

BATCH_SIZE = 1

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch


if opt.marc_data_train is not None:
    video_list = 'test_data/test_marc_video_list_' + opt.marc_data_train + '.txt'
else:
    video_list = 'test_data/test_davis_video_list.txt'


eval_num_threads = 2
video_data_loader = aligned_data_loader.DAVISDataLoader(video_list, BATCH_SIZE)
video_dataset = video_data_loader.load_data()
print('========================= Video dataset #images = %d =========' %
      len(video_data_loader))

model = pix2pix_model.Pix2PixModel(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch = 0
global_step = 0


# make dict with data frames #TODO this is inefficient and doesn't use the dataloader intelligently
data_frames = {}
for i, data in enumerate(video_dataset):
    data_frames[i] = data

for i in range(0, len(data_frames) - 1, 1):
    # for i in range(len(data_frames) - 2, 0, -1): #backwards

    first_data = data_frames[i]
    second_data = data_frames[i + 1]

    inputs = []

    stacked_img_1 = first_data[0]
    # don't actually need this for train I think, but need it to make the forward function happy (could change)
    targets_1 = first_data[1]
    stacked_img_2 = second_data[0]
    targets_2 = second_data[1]

    inputs = [stacked_img_1, stacked_img_2]
    targets = [targets_1, targets_2]

    model.latent_train(inputs, targets)

# torch.save(model.netG.cpu().state_dict(), 'latent_constrained_model.pth')
# model.save_network(model.netG, 'G', "latent_constrained", model.gpu_ids) #using their .save_network() function
# torch.save(model.netG.module.cpu().state_dict(), 'checkpoints/test_local/'+str(time.time())+'latent_constrained_model_net_G.pth')
torch.save(model.netG.module.cpu().state_dict(),
           'checkpoints/test_local/' + opt.save_weights + '_net_G.pth')
