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

import sys
import time
from os import system
import torch
from options.train_options import TrainOptions
from loaders import aligned_data_loader
from models import pix2pix_model

BATCH_SIZE = 2  # number of images to load in simultaneously from data loader

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

############
############
# old way to load data and train

# make dict with data frames #TODO this is inefficient and doesn't use the dataloader intelligently
# data_frames = {}
# for i, data in enumerate(video_dataset):
#     data_frames[i] = data

# for i in range(0, len(data_frames) - 1, 1):
#     # for i in range(len(data_frames) - 2, 0, -1): #backwards

#     first_data = data_frames[i]
#     second_data = data_frames[i + 1]

#     inputs = []

#     stacked_img_1 = first_data[0]
#     # don't actually need this for train I think, but need it to make the forward function happy (could change)
#     targets_1 = first_data[1]
#     stacked_img_2 = second_data[0]
#     targets_2 = second_data[1]

#     inputs = [stacked_img_1, stacked_img_2]
#     targets = [targets_1, targets_2]
#     # print("targets", targets)
# # sys.exit()

#     model.latent_train(inputs, targets)

############
############

prev_frame = None
prev_target = None
for i, data in enumerate(video_dataset):
    # assume we have new pair (b, c), where prev frame was a

    # first, constrain curr frame with prev frame (a, b)
    img, target = data
    img_1 = img[0, :, :, :].unsqueeze(0)
    targets_list = target['img_1_path']
    target_1 = {'img_1_path': [targets_list[0]]}
    if(prev_frame is not None):
        if opt.constrain_output:
            model.output_train([prev_frame, img_1], [prev_target, target_1], opt.L1_loss)
        else:
            model.latent_train([prev_frame, img_1], [prev_target, target_1], opt.L1_loss)

    # then constrain (b, c)
    if (img.shape[0] > 1):  # if there is actually a full batch
        img_2 = img[1, :, :, :].unsqueeze(0)
        target_2 = {'img_1_path': [targets_list[1]]}
        targets = [target_1, target_2]
        if opt.constrain_output:
             model.output_train([img_1, img_2], targets, opt.L1_loss)
        else:
             model.latent_train([img_1, img_2], targets, opt.L1_loss)
        prev_target = target_2  # set c to the prev, so it will be used for next constraint
        prev_frame = img_2
    else:
        break


# model.save_network(model.netG, 'G', "latent_constrained", model.gpu_ids) #using their .save_network() function
# torch.save(model.netG.module.cpu().state_dict(), 'checkpoints/test_local/'+str(time.time())+'latent_constrained_model_net_G.pth')
save_weights = opt.save_weights
if save_weights is None:
    save_weights = str(time.time())+'latent_constrained_model'
torch.save(model.netG.module.cpu().state_dict(),
           'checkpoints/test_local/' + save_weights + '_net_G.pth')
