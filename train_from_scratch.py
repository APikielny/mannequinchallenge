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
from models import networks
import cv2
import torch.multiprocessing

BATCH_SIZE = 4  # number of images to load in simultaneously from data loader

# import gc
# gc.collect()
# torch.cuda.empty_cache()

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

# video_list = 'test_data/supervision_list.txt'

torch.multiprocessing.set_sharing_strategy('file_system')
# video_list = 'test_data/single_pair_2.txt' #for viewing masks
# video_list = 'test_data/full_train_list_grid.txt'
video_list = 'test_data/small_train_list_grid.txt'
test_video_list = 'test_data/test_list_grid.txt'


eval_num_threads = 2
# video_data_loader = aligned_data_loader.DAVISDataLoader(video_list, BATCH_SIZE)
# video_dataset = video_data_loader.load_data()
video_data_loader = aligned_data_loader.SupervisionDataLoader(video_list, BATCH_SIZE)
video_dataset = video_data_loader.load_data()
print('========================= Video dataset #images = %d =========' %
      len(video_data_loader) * BATCH_SIZE)

model = pix2pix_model.Pix2PixModel(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch = 0
global_step = 0

######################################################
## goal: sample random consecutive frames, optimize based on their depths and the latent constraint between them
######################################################

#randomly initialize weights
# networks.init_weights(model.netG)

# img, target = data_frames[0]
# img_1 = img[0, :, :, :].unsqueeze(0)
# targets_list = target['img_1_path']
# target_1 = {'img_1_path': [targets_list[0]]}

# gt_depth = target['depth_gt']
# gt_depth = gt_depth.cuda()
# print(type(gt_depth))
# print(gt_depth.size())

# depth_path = '/home/adam/Desktop/repos/adam-mannequinchallenge/test_data/scratch_training_test/frame28_img.jpg'
# depth_img = cv2.imread(depth_path, 0) #read as greyscale
# torch_depth = torch.from_numpy(depth_img).cuda()
# print("depth input shape: ", torch_depth.shape)

# old
# data_frames = {}
# for i, data in enumerate(video_dataset):
#     data_frames[i] = data

# num_frames = len(data_frames)
# print("number of frames: ", num_frames)

# Doesn't support larger batch size currently
# Switched to passing in target, so we have access to gt_mask
# max_epochs = 1
# for epoch in range(max_epochs):
#     for i in range(num_frames):
#         img, target = data_frames[i]
#         model.depth_train(img, target)

# new
max_epochs = 10
num_batches = len(video_data_loader)
print("Total number of batches: ", num_batches)

save_interim_models = True
save_weights = opt.save_weights
if save_weights is None:
    save_weights = str(time.time())+'train_from_scratch_model'

for epoch in range(max_epochs):
    for i, data in enumerate(video_dataset):
        img, target = data
        # model.depth_train(img, target)
        print("Batch index - ", i, " Epoch - ", epoch)
        model.depth_train(i, img, target, num_batches)
        # model.depth_and_latent_train_v2(i, img, target, num_batches)
        
        ########
        # examining masks vs. depth values:
        #######
        # cv2.imwrite('test_data/scratch_debug_masks/mask0.3.png', target['gt_mask'][0].detach().numpy()*255)
        # cv2.imwrite('test_data/scratch_debug_masks/depth0.3.png', target['depth_gt'][0].detach().numpy()*255)
        # exit()
    
    #TODO
    #instead of saving interim models, can just run an evaluating/test script on the current model and save it somewhere. Would save a step.
    if save_interim_models:
        # print("Saving interim model to ", '/data/jhtlab/apikieln/checkpoints/test_local/' + save_weights + "_epoch_" + str(epoch) + '_net_G.pth')
        # torch.save(model.netG.module.cpu().state_dict(),
        #    '/data/jhtlab/apikieln/checkpoints/test_local/' + save_weights + "_epoch_" + str(epoch) + '_net_G.pth')
        model.switch_to_eval()
        save_path = 'test_data/viz_predictions/'
        self.weights = self.save_weights+"_epoch_"+str(epoch) #hacky way to get proper path to save images to
        print('save_path %s' % save_path)
        
        print("Testing current model.")
        test_video_data_loader = aligned_data_loader.DAVISDataLoader(test_video_list, BATCH_SIZE)
        test_video_dataset = test_video_data_loader.load_data()

        for j, data_test in enumerate(test_video_dataset):
            print(j)
            stacked_img_test = data_test[0]
            targets_test = data_test[1]
            model.run_and_save_DAVIS(stacked_img_test, targets_test, save_path, opt.visualize)

        print("Switching back to train.")
        model.switch_to_train()

print("Finished training. ")

torch.save(model.netG.module.cpu().state_dict(),
           '/data/jhtlab/apikieln/checkpoints/test_local/' + save_weights + '_net_G.pth')

print("Saved to ", '/data/jhtlab/apikieln/checkpoints/test_local/' + save_weights + '_net_G.pth')

# python train_from_scratch.py --lr 0.0001 --save_weights overfit-small-set-0.0001
# python test_davis_videos.py --weights overfit-small-set-0.0001
