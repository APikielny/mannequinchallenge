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
from plot_train_losses import plot_losses

# import gc
# gc.collect()
# torch.cuda.empty_cache()

opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

BATCH_SIZE = opt.batch_size  # number of images to load in simultaneously from data loader
assert ((BATCH_SIZE == 16) or (BATCH_SIZE == 8) or (BATCH_SIZE == 4) or (BATCH_SIZE == 2) or (BATCH_SIZE == 1))
k = 16/BATCH_SIZE


def save_interim_results_func(epoch_num):
    # print("Saving interim model to ", '/data/jhtlab/apikieln/checkpoints/test_local/' + save_weights + "_epoch_" + str(epoch) + '_net_G.pth')
    # torch.save(model.netG.module.cpu().state_dict(),
    #    '/data/jhtlab/apikieln/checkpoints/test_local/' + save_weights + "_epoch_" + str(epoch) + '_net_G.pth')
    model.switch_to_eval()
    save_path = 'test_data/viz_predictions/'
    weights = opt.save_weights+"/epoch_"+str(epoch_num) #hacky way to get proper path to save images to
    print('save_path %s' % save_path)
    
    print("Testing current model.")
    test_video_data_loader = aligned_data_loader.DAVISDataLoader(test_video_list, BATCH_SIZE)
    test_video_dataset = test_video_data_loader.load_data()

    for j, data_test in enumerate(test_video_dataset):
        print(j)
        stacked_img_test = data_test[0]
        targets_test = data_test[1]
        model.run_and_save_DAVIS_interim(stacked_img_test, targets_test, save_path, opt.visualize, weights)

    print("Switching back to train.")
    model.switch_to_train()

# video_list = 'test_data/supervision_list.txt'
torch.multiprocessing.set_sharing_strategy('file_system')
# video_list = 'test_data/single_pair_2.txt' #for viewing masks
video_list = 'test_data/full_train_list_grid.txt'
# video_list = 'test_data/small_train_list_grid.txt'
# test_video_list = 'test_data/test_list_grid.txt'
test_video_list = 'test_data/test_list_grid_adam_translate.txt'

#overfit:
# video_list = 'test_data/small_train_list_grid.txt'
# test_video_list = 'test_data/small_test_list_grid.txt'



eval_num_threads = 2
# video_data_loader = aligned_data_loader.DAVISDataLoader(video_list, BATCH_SIZE)
# video_dataset = video_data_loader.load_data()
video_data_loader = aligned_data_loader.SupervisionDataLoader(video_list, BATCH_SIZE)
video_dataset = video_data_loader.load_data()
print('========================= Video dataset #images = %d =========' %
      len(video_data_loader) * BATCH_SIZE)

model = pix2pix_model.Pix2PixModel(opt)#, True) TODO CHANGE BACK TO TRUE FOR TRAIN FROM SCRATCH

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch = 0
global_step = 0

######################################################
## goal: sample random consecutive frames, optimize based on their depths and the latent constraint between them
######################################################

#randomly initialize weights
# networks.init_weights(model.netG)
max_epochs = opt.epochs
num_batches = len(video_data_loader)
print("Total number of batches: ", num_batches)

save_interim_results = True
save_weights = opt.save_weights
if save_weights is None:
    save_weights = str(time.time())+'train_from_scratch_model'

latent_loss_list = []
supervision_loss_list = []

if save_interim_results:
    save_interim_results_func("pre_train")

for epoch in range(max_epochs):
    latent_loss_accum = 0
    supervision_loss_accum = 0
    counter = 0 #do I need this? can I just use i?
    for i, data in enumerate(video_dataset):
        img, target = data
        # model.depth_train(img, target)
        print("Batch index - ", i, " Epoch - ", epoch)
        # model.depth_train(i, img, target, num_batches)
        latent_loss, supervision_loss = model.depth_and_latent_train_v2(i, img, target, num_batches, k)
        
        if (i%10000 == 0):
            counter += 1
            latent_loss_accum += latent_loss
            supervision_loss_accum += supervision_loss
        ########
        # examining masks vs. depth values:
        #######
        # cv2.imwrite('test_data/scratch_debug_masks/mask0.3.png', target['gt_mask'][0].detach().numpy()*255)
        # cv2.imwrite('test_data/scratch_debug_masks/depth0.3.png', target['depth_gt'][0].detach().numpy()*255)
        # exit()
    latent_loss_list.append(latent_loss_accum/i)
    supervision_loss_list.append(supervision_loss_accum/i)

    #TODO
    #instead of saving interim models, can just run an evaluating/test script on the current model and save it somewhere. Would save a step.
    if save_interim_results:
        save_interim_results_func(epoch)

if opt.plot_losses:
    #plot losses
    plot_losses(latent_loss_list, supervision_loss_list, opt.save_weights, opt.latent_weight)

print("Finished training. ")
# model.switch_to_eval()
# model.run_and_save_DAVIS_interim(stacked_img_test, targets_test, save_path, opt.visualize, opt.save_weights+"/final")
# model.switch_to_train()
save_interim_results_func("final")

torch.save(model.netG.module.cpu().state_dict(),
           '/data/jhtlab/apikieln/checkpoints/test_local/' + save_weights + '_net_G.pth')

print("Saved to ", '/data/jhtlab/apikieln/checkpoints/test_local/' + save_weights + '_net_G.pth')

# python train_from_scratch.py --lr 0.0001 --save_weights overfit-small-set-0.0001
# python test_davis_videos.py --weights overfit-small-set-0.0001
