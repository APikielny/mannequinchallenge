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

import torch
from options.train_options import TrainOptions
from loaders import aligned_data_loader
from models import pix2pix_model

BATCH_SIZE = 1

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

# video_list = 'test_data/test_davis_video_list.txt'
video_list = 'test_data/test_marc_video_list.txt'

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


#make dict with data frames
data_frames = {}
for i,data in enumerate(video_dataset):
    data_frames[i] = data


print("len", len(data_frames))




for i in range(0, len(data_frames), 2):


    first_data = data_frames[i]
    second_data = data_frames[i + 1]


    inputs = []

    stacked_img_1 = first_data[0]
    targets_1 = first_data[1] #don't actually need this for train I think, but need it to make the forward function happy (could change)
    stacked_img_2 = second_data[0]
    targets_2 = second_data[1]

    inputs = [stacked_img_1, stacked_img_2]
    targets = [targets_1, targets_2]

    model.latent_train(inputs, targets)

# print(
#     '=================================  BEGIN VALIDATION ====================================='
# )

# print('TESTING ON VIDEO')

# model.switch_to_eval()
# save_path = 'test_data/viz_predictions/'
# print('save_path %s' % save_path)

# for i, data in enumerate(video_dataset):
#     # print(i)
#     stacked_img = data[0]
#     targets = data[1]
#     model.run_and_save_DAVIS(stacked_img, targets, save_path)
