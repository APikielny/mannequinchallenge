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

from dis import dis
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import os
from models import base_model
from models import networks
import sys
import h5py
import os.path
from skimage.io import imsave
from models import hourglass
from models import loss


import torchvision.utils as vutils

# torch.manual_seed(1)
class HourglassVariant(torch.nn.Module):

    def __init__(self, num_input, model):
        super(HourglassVariant, self).__init__()
        layer_list = list(model.children())
        removed = layer_list[1:-1]
        self.pred_layer = layer_list[-1]
        model = torch.nn.Sequential(*removed)

        uncertainty_layer = [
            torch.nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1)),
            torch.nn.Sigmoid()
        ]
        self.uncertainty_layer = torch.nn.Sequential(*uncertainty_layer)
        new_input_layer = torch.nn.Conv2d(
            num_input, 128, (7, 7), (1, 1), (3, 3))

        nn.init.normal_(new_input_layer.weight.data, 0.0, 0.02)
        nn.init.constant_(new_input_layer.bias.data, 0.0)

        self.new_model = torch.nn.Sequential(new_input_layer, model)

    def forward(self, input_):
        pred_feature = self.new_model(input_)

        pred_d = self.pred_layer(pred_feature)
        pred_confidence = self.uncertainty_layer(pred_feature)

        return pred_d, pred_confidence


class Pix2PixModel(base_model.BaseModel):

    def name(self):
        return 'Pix2PixModel'

    def __init__(self, opt, _isTrain=False):
        self.initialize(opt)
        self.weights = opt.weights
        self.latent_constraint_weight = 1e-4

        self.mode = opt.mode
        if opt.input == 'single_view':
            self.num_input = 3
        elif opt.input == 'two_view':
            self.num_input = 6
        elif opt.input == 'two_view_k':
            self.num_input = 7
        else:
            raise ValueError("Unknown input type %s" % opt.input)

        if self.mode == 'Ours_Bilinear':
            print(
                '======================================  DIW NETWORK TRAIN FROM %s======================='
                % self.mode)

            new_model = hourglass.HourglassModel(self.num_input)

            print(
                '===================Loading Pretrained Model OURS ==================================='
            )

            if opt.weights is not None:
                model_parameters = self.load_network(
                    new_model, 'G', opt.weights)
            else:
                if not _isTrain:
                    if self.num_input == 7:
                        model_parameters = self.load_network(
                            new_model, 'G', 'best_depth_Ours_Bilinear_inc_7')
                    elif self.num_input == 3:
                        model_parameters = self.load_network(
                            new_model, 'G', 'best_depth_Ours_Bilinear_inc_3')
                    elif self.num_input == 6:
                        model_parameters = self.load_network(
                            new_model, 'G', 'best_depth_Ours_Bilinear_inc_6')
                    else:
                        print('Something Wrong')
                        sys.exit()

            # TODO for some reason have to move this above parallel depending on my model vs. original models
            new_model.load_state_dict(model_parameters)

            new_model = torch.nn.parallel.DataParallel(
                new_model.cuda(), device_ids=range(torch.cuda.device_count()))

            self.netG = new_model

        else:
            print('ONLY SUPPORT Ours_Bilinear')
            sys.exit()

        self.old_lr = opt.lr
        self.netG.train()

        if True:
            self.criterion_joint = networks.JointLoss(opt)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer_G, opt)
            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            print('-----------------------------------------------')

    def set_writer(self, writer):
        self.writer = writer

    def set_input(self, stack_imgs, targets):
        self.input = stack_imgs
        self.targets = targets

    def forward(self):

        # run first network
        self.input_images = autograd.Variable(
            self.input.cuda(), requires_grad=False)
        human_mask = 1.0 - autograd.Variable(
            self.targets['env_mask'].cuda(), requires_grad=False).unsqueeze(1)
        keypoints_img = autograd.Variable(
            self.targets['keypoints_img'].cuda(), requires_grad=False).unsqueeze(1)

        # stack inputs
        stack_inputs = None

        if self.num_input == 7:
            input_log_depth = autograd.Variable(
                self.targets['input_log_depth'].cuda(),
                requires_grad=False).unsqueeze(1)
            input_confidence = autograd.Variable(
                self.targets['input_confidence'].cuda(),
                requires_grad=False).unsqueeze(1)
            stack_inputs = torch.cat((self.input_images, human_mask, keypoints_img,
                                      input_log_depth, input_confidence), 1)
        elif self.num_input == 6:
            input_log_depth = autograd.Variable(
                self.targets['input_log_depth'].cuda(),
                requires_grad=False).unsqueeze(1)
            input_confidence = autograd.Variable(
                self.targets['input_confidence'].cuda(),
                requires_grad=False).unsqueeze(1)
            stack_inputs = torch.cat(
                (self.input_images, human_mask, input_log_depth, input_confidence), 1)
        elif self.num_input == 3:
            stack_inputs = self.input_images
        else:
            print('SOMETHING WRONG with num_input !!!!!!!!!!!!!!!!!!!!!!!')
            sys.exit()

        self.prediction_d, self.pred_confidence = self.netG.forward(
            stack_inputs)
        self.prediction_d = self.prediction_d.squeeze(1)
        self.pred_confidence = self.pred_confidence.squeeze(1)

    def get_image_paths(self):
        return self.image_paths

    def write_summary(self,
                      mode_name,
                      input_images,
                      prediction_d,
                      pred_confidence,
                      targets,
                      n_iter,
                      loss=None):

        invere_depth_pred = torch.exp(-prediction_d.data.cpu()).unsqueeze(1).repeat(
            1, 3, 1, 1)

        invere_depth_gt = 1.0 / \
            targets['depth_gt'].unsqueeze(1).repeat(1, 3, 1, 1)
        gt_mask = targets['gt_mask'].unsqueeze(1).repeat(1, 3, 1, 1)

        invere_depth_gt = invere_depth_gt * gt_mask
        min_depth, max_depth = np.percentile(
            invere_depth_pred.numpy(), [1, 99])
        invere_depth_pred[invere_depth_pred > max_depth] = 0.0
        invere_depth_pred[invere_depth_pred < min_depth] = 0.0

        inv_depth_mask = invere_depth_pred * gt_mask

        human_mask = 1.0 - targets['env_mask'].unsqueeze(1).repeat(1, 3, 1, 1)
        input_confidence = targets['input_confidence'].unsqueeze(1).repeat(
            1, 3, 1, 1)
        pred_confidence_saved = pred_confidence.data.unsqueeze(
            1).repeat(1, 3, 1, 1)

        if loss:
            self.writer.add_scalar(mode_name + '/loss', loss, n_iter)

        self.writer.add_image(
            mode_name + '/image',
            vutils.make_grid(
                input_images[:8, :, :, :].data.cpu(), normalize=True),
            n_iter)
        self.writer.add_image(
            mode_name + '/pred_full',
            vutils.make_grid(invere_depth_pred[:8, :, :, :], normalize=True),
            n_iter)
        self.writer.add_image(
            mode_name + '/pred_mask',
            vutils.make_grid(inv_depth_mask[:8, :, :, :], normalize=True), n_iter)
        self.writer.add_image(
            mode_name + '/pred_confidence',
            vutils.make_grid(
                pred_confidence_saved[:8, :, :, :], normalize=True),
            n_iter)

        self.writer.add_image(
            mode_name + '/gt_depth',
            vutils.make_grid(invere_depth_gt[:8, :, :, :], normalize=True), n_iter)
        self.writer.add_image(
            mode_name + '/gt_mask',
            vutils.make_grid(gt_mask[:8, :, :, :], normalize=True), n_iter)

        self.writer.add_image(
            mode_name + '/human_mask',
            vutils.make_grid(human_mask[:8, :, :, :], normalize=True), n_iter)
        self.writer.add_image(
            mode_name + '/input_confidence',
            vutils.make_grid(input_confidence[:8, :, :, :], normalize=True), n_iter)

    def backward_G(self, n_iter):
        # Combined loss
        self.loss_joint = self.criterion_joint(self.input_images, self.prediction_d, None, self.targets)
        print('Train loss is %f ' % self.loss_joint)

        # add to tensorboard
        if n_iter % 100 == 0:
            self.write_summary('Train', self.input_images, self.prediction_d,
                               self.pred_confidence, self.targets, n_iter,
                               self.loss_joint)

        self.loss_joint_var = self.criterion_joint.get_loss_var()
        # print(type(self.loss_joint_var), self.loss_joint_var.size())
        self.loss_joint_var.backward()

    def optimize_parameters(self, n_iter):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G(n_iter)
        self.optimizer_G.step()

    def evlaute_M_error(self, input_, targets, n_iter, write_to_summary):
        # switch to evaluation mode
        input_imgs = autograd.Variable(input_.cuda(), requires_grad=False)
        # stack inputs
        human_mask = 1.0 - autograd.Variable(
            targets['env_mask'].cuda(), requires_grad=False).unsqueeze(1)
        keypoints_img = autograd.Variable(
            targets['keypoints_img'].cuda(), requires_grad=False).unsqueeze(1)

        if self.num_input == 7:
            input_log_depth = autograd.Variable(
                targets['input_log_depth'].cuda(), requires_grad=False).unsqueeze(1)
            input_confidence = autograd.Variable(
                targets['input_confidence'].cuda(), requires_grad=False).unsqueeze(1)
            stack_inputs = torch.cat((input_imgs, human_mask, keypoints_img,
                                      input_log_depth, input_confidence), 1)
        elif self.num_input == 6:
            input_log_depth = autograd.Variable(
                targets['input_log_depth'].cuda(), requires_grad=False).unsqueeze(1)
            input_confidence = autograd.Variable(
                targets['input_confidence'].cuda(), requires_grad=False).unsqueeze(1)
            stack_inputs = torch.cat(
                (input_imgs, human_mask, input_log_depth, input_confidence), 1)
        elif self.num_input == 3:
            stack_inputs = input_imgs
        else:
            print('SOMETHING WRONG!!!!!!!!!!!!!!!!!!!!!!!')
            sys.exit()

        prediction_d, _ = self.netG.forward(stack_inputs)

        pred_log_d = prediction_d.squeeze(1)

        sc_inv_full, sc_inv_human, sc_inv_env, sc_inv_intra, sc_inv_inter = self.criterion_joint.compute_si_rmse(
            pred_log_d.data, targets)
        sc_inv_errors = [
            sc_inv_full.item(),
            sc_inv_human.item(),
            sc_inv_env.item(),
            sc_inv_intra.item(),
            sc_inv_inter.item()
        ]

        return sc_inv_errors

    def eval_save_tum_img(self, input_, targets, save_path):
        input_imgs = autograd.Variable(input_.cuda())
        human_mask = 1.0 - autograd.Variable(
            targets['env_mask'].cuda(), requires_grad=False).unsqueeze(1)
        keypoints_img = autograd.Variable(
            targets['keypoints_img'].cuda(), requires_grad=False).unsqueeze(1)

        if self.num_input == 7:
            input_log_depth = autograd.Variable(
                targets['input_log_depth'].cuda(), requires_grad=False).unsqueeze(1)
            input_confidence = autograd.Variable(
                targets['input_confidence'].cuda(), requires_grad=False).unsqueeze(1)
            stack_inputs = torch.cat((input_imgs, human_mask, keypoints_img,
                                      input_log_depth, input_confidence), 1)
        elif self.num_input == 6:
            input_log_depth = autograd.Variable(
                targets['input_log_depth'].cuda(), requires_grad=False).unsqueeze(1)
            input_confidence = autograd.Variable(
                targets['input_confidence'].cuda(), requires_grad=False).unsqueeze(1)
            stack_inputs = torch.cat(
                (input_imgs, human_mask, input_log_depth, input_confidence), 1)
        elif self.num_input == 3:
            stack_inputs = input_imgs
        else:
            print('SOMETHING WRONG!!!!!!!!!!!!!!!!!!!!!!!')
            sys.exit()

        prediction_d, pred_confidence = self.netG.forward(stack_inputs)
        prediction_d = torch.exp(prediction_d.squeeze(1))
        pred_confidence = pred_confidence.squeeze(1)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(0, len(targets['img_1_path'])):

            youtube_dir = save_path + targets['img_1_path'][i].split('/')[-2]

            if not os.path.exists(youtube_dir):
                os.makedirs(youtube_dir)

            pred_d_ref = prediction_d.data[i, :, :].cpu().numpy()
            saved_img = np.transpose(
                input_imgs[i, :, :, :].cpu().numpy(), (1, 2, 0))

            output_path = youtube_dir + '/' + \
                targets['img_1_path'][i].split('/')[-1]

            print('output_path', output_path)
            input_confidence = targets['input_confidence'][i]
            gt_depth = targets['depth_gt'][i]
            gt_mask = targets['gt_mask'][i]
            human_mask = 1.0 - targets['env_mask'][i]

            # K = targets['K'][i]
            # T_1_G = targets['T_1_G'][i]

            hdf5_file_write = h5py.File(output_path, 'w')
            hdf5_file_write.create_dataset(
                '/prediction/img', data=saved_img, dtype='float32')
            hdf5_file_write.create_dataset(
                '/prediction/pred_depth', data=pred_d_ref, dtype='float32')
            hdf5_file_write.create_dataset(
                '/prediction/gt_depth', data=gt_depth, dtype='float32')
            hdf5_file_write.create_dataset(
                '/prediction/gt_mask', data=gt_mask, dtype='float32')
            hdf5_file_write.create_dataset(
                '/prediction/input_confidence',
                data=input_confidence,
                dtype='float32')
            hdf5_file_write.create_dataset(
                '/prediction/human_mask', data=human_mask, dtype='float32')

    def evaluate_tum_error(self, input_, targets, n_iter, write_to_summary):
        input_imgs = autograd.Variable(input_.cuda(), requires_grad=False)
        human_mask = 1.0 - autograd.Variable(
            targets['env_mask'].cuda(), requires_grad=False).unsqueeze(1)
        keypoints_img = autograd.Variable(
            targets['keypoints_img'].cuda(), requires_grad=False).unsqueeze(1)

        if self.num_input == 7:
            input_log_depth = autograd.Variable(
                targets['input_log_depth'].cuda(), requires_grad=False).unsqueeze(1)
            input_confidence = autograd.Variable(
                targets['input_confidence'].cuda(), requires_grad=False).unsqueeze(1)
            stack_inputs = torch.cat((input_imgs, human_mask, keypoints_img,
                                      input_log_depth, input_confidence), 1)
        elif self.num_input == 6:
            input_log_depth = autograd.Variable(
                targets['input_log_depth'].cuda(), requires_grad=False).unsqueeze(1)
            input_confidence = autograd.Variable(
                targets['input_confidence'].cuda(), requires_grad=False).unsqueeze(1)
            stack_inputs = torch.cat(
                (input_imgs, human_mask, input_log_depth, input_confidence), 1)
        elif self.num_input == 3:
            stack_inputs = input_imgs
        else:
            print('SOMETHING WRONG!!!!!!!!!!!!!!!!!!!!!!!')
            sys.exit()

        prediction_d, _ = self.netG.forward(stack_inputs)

        pred_log_d = prediction_d.squeeze(1)
        pred_d = torch.exp(pred_log_d)

        sc_inv_full, sc_inv_human, sc_inv_env, sc_inv_intra, sc_inv_inter = self.criterion_joint.compute_si_rmse(
            pred_log_d.data, targets)
        l1_rel_full = self.criterion_joint.compute_l1_rel_error(
            pred_d.data, targets)
        RMSE_full = self.criterion_joint.compute_rmse_error(
            pred_d.data, targets)

        sc_inv_errors = [
            sc_inv_full.item(),
            sc_inv_human.item(),
            sc_inv_env.item(),
            sc_inv_intra.item(),
            sc_inv_inter.item()
        ]

        return sc_inv_errors, l1_rel_full, RMSE_full

    def eval_save_img(self, input_, targets, save_path):
        input_imgs = autograd.Variable(input_.cuda(), requires_grad=False)
        human_mask = 1.0 - autograd.Variable(
            targets['env_mask'].cuda(), requires_grad=False).unsqueeze(1)
        keypoints_img = autograd.Variable(
            targets['keypoints_img'].cuda(), requires_grad=False).unsqueeze(1)

        if self.num_input == 7:
            input_log_depth = autograd.Variable(
                targets['input_log_depth'].cuda(), requires_grad=False).unsqueeze(1)
            input_confidence = autograd.Variable(
                targets['input_confidence'].cuda(), requires_grad=False).unsqueeze(1)
            stack_inputs = torch.cat((input_imgs, human_mask, keypoints_img,
                                      input_log_depth, input_confidence), 1)
        elif self.num_input == 6:
            human_mask = 1.0 - autograd.Variable(
                targets['env_mask'].cuda(), requires_grad=False).unsqueeze(1)
            input_log_depth = autograd.Variable(
                targets['input_log_depth'].cuda(), requires_grad=False).unsqueeze(1)
            input_confidence = autograd.Variable(
                targets['input_confidence'].cuda(), requires_grad=False).unsqueeze(1)
            stack_inputs = torch.cat(
                (input_imgs, human_mask, input_log_depth, input_confidence), 1)
        elif self.num_input == 3:
            stack_inputs = input_imgs
        else:
            print('SOMETHING WRONG!!!!!!!!!!!!!!!!!!!!!!!')
            sys.exit()

        prediction_log_d, _ = self.netG.forward(stack_inputs)
        prediction_d = torch.exp(prediction_log_d.squeeze(1))

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(0, len(targets['img_1_path'])):

            youtube_dir = save_path + targets['img_1_path'][i].split('/')[-2]

            print('youtube_dir ', youtube_dir)

            if not os.path.exists(youtube_dir):
                os.makedirs(youtube_dir)

            pred_d_ref = prediction_d.data[i, :, :].cpu().numpy()
            saved_img = np.transpose(
                input_imgs[i, :, :, :].cpu().numpy(), (1, 2, 0))
            human_mask_ref = human_mask.data[i, 0, :, :].cpu().numpy()

            output_path = youtube_dir + '/' + \
                targets['img_1_path'][i].split('/')[-1]
            gt_depth_ref = targets['depth_gt'][i]
            gt_mask_ref = targets['gt_mask'][i]
            input_confidence_ref = targets['input_confidence'][i]
            keypoints_img = targets['keypoints_img'][i]

            input_depth_final_ref = targets['input_depth_final'][i]
            full_flow = targets['full_flow'][i]
            img_2 = targets['img_2'][i]

            K = targets['K'][i]
            T_1_G = targets['T_1_G'][i]

            hdf5_file_write = h5py.File(output_path, 'w')
            hdf5_file_write.create_dataset('/prediction/img', data=saved_img)
            hdf5_file_write.create_dataset(
                '/prediction/pred_depth', data=pred_d_ref)

            hdf5_file_write.create_dataset(
                '/prediction/gt_depth', data=gt_depth_ref)
            hdf5_file_write.create_dataset(
                '/prediction/gt_mask', data=gt_mask_ref)
            hdf5_file_write.create_dataset(
                '/prediction/human_mask', data=human_mask_ref)
            hdf5_file_write.create_dataset(
                '/prediction/input_confidence', data=input_confidence_ref)
            hdf5_file_write.create_dataset(
                '/prediction/input_pp_depth', data=input_depth_final_ref)
            hdf5_file_write.create_dataset(
                '/prediction/keypoints_img', data=keypoints_img)
            hdf5_file_write.create_dataset(
                '/prediction/full_flow', data=full_flow)
            hdf5_file_write.create_dataset('/prediction/img_2', data=img_2)

            hdf5_file_write.create_dataset('/prediction/K', data=K)
            hdf5_file_write.create_dataset('/prediction/T_1_G', data=T_1_G)
            hdf5_file_write.close()

    def run_and_save_videos_prediction(self, input_, targets, save_path):
        input_imgs = autograd.Variable(input_.cuda(), requires_grad=False)

        human_mask = 1.0 - autograd.Variable(
            targets['env_mask'].cuda(), requires_grad=False).unsqueeze(1)
        input_log_depth = autograd.Variable(
            targets['input_log_depth'].cuda(), requires_grad=False).unsqueeze(1)
        input_confidence = autograd.Variable(
            targets['input_confidence'].cuda(), requires_grad=False).unsqueeze(1)
        keypoints_img = autograd.Variable(
            targets['keypoints_img'].cuda(), requires_grad=False).unsqueeze(1)

        mvs_depth = autograd.Variable(
            targets['mvs_depth'].cuda(), requires_grad=False)
        input_depth = autograd.Variable(
            targets['input_depth'].cuda(), requires_grad=False)

        full_flow = targets['full_flow']

        if self.num_input == 7:
            stack_inputs = torch.cat((input_imgs, human_mask, keypoints_img,
                                      input_log_depth, input_confidence), 1)
        elif self.num_input == 6:
            stack_inputs = torch.cat(
                (input_imgs, human_mask, input_log_depth, input_confidence), 1)
        elif self.num_input == 3:
            stack_inputs = input_imgs
        else:
            print('SOMETHING WRONG!!!!!!!!!!!!!!!!!!!!!!!')
            sys.exit()

        prediction_d, pred_confidence = self.netG.forward(stack_inputs)

        pred_log_d = prediction_d.squeeze(1)
        pred_d = torch.exp(pred_log_d)
        pred_confidence = pred_confidence.squeeze(1)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(0, len(targets['img_1_path'])):

            youtube_dir = save_path + targets['img_1_path'][i].split('/')[-2]

            if not os.path.exists(youtube_dir):
                os.makedirs(youtube_dir)

            saved_img = np.transpose(
                input_imgs[i, :, :, :].cpu().numpy(), (1, 2, 0))

            pred_d_ref = pred_d.data[i, :, :].cpu().numpy()
            pred_confidence_ref = pred_confidence.data[i, :, :].cpu().numpy()
            human_mask_ref = human_mask.data[i, 0, :, :].cpu().numpy()

            mvs_depth_ref = mvs_depth.data[i, :, :].cpu().numpy()

            input_depth_ref = input_depth[i, :, :].data.cpu().numpy()
            input_confidence_ref = input_confidence[i, 0, :, :].data.cpu(
            ).numpy()
            full_flow_ref = full_flow[i, :, :, :].data.cpu().numpy()

            output_path = youtube_dir + '/' + \
                targets['img_1_path'][i].split('/')[-1]

            K = targets['K'][i]
            T_1_G = targets['T_1_G'][i]
            original_mvs_depth = targets['original_mvs_depth'][i]

            print('output_path', output_path)
            hdf5_file_write = h5py.File(output_path, 'w')
            hdf5_file_write.create_dataset('/prediction/img', data=saved_img)
            hdf5_file_write.create_dataset(
                '/prediction/pred_depth', data=pred_d_ref)
            hdf5_file_write.create_dataset('/prediction/K', data=K)
            hdf5_file_write.create_dataset('/prediction/T_1_G', data=T_1_G)
            hdf5_file_write.create_dataset(
                '/prediction/confidence', data=pred_confidence_ref)
            hdf5_file_write.create_dataset(
                '/prediction/human_mask', data=human_mask_ref)
            hdf5_file_write.create_dataset(
                '/prediction/input_depth', data=input_depth_ref)
            hdf5_file_write.create_dataset(
                '/prediction/input_confidence', data=input_confidence_ref)
            hdf5_file_write.create_dataset(
                '/prediction/mvs_depth', data=mvs_depth_ref)
            hdf5_file_write.create_dataset(
                '/prediction/full_flow', data=full_flow_ref)
            hdf5_file_write.create_dataset(
                '/prediction/original_mvs_depth', data=original_mvs_depth)

            hdf5_file_write.close()

    def get_latent(self, input, targets):
        input_imgs = autograd.Variable(input.cuda(), requires_grad=True)
        stack_inputs = input_imgs
        latent = self.netG.forward(stack_inputs, targets, False, True)
        # pred_log_d = prediction_d.squeeze(1)
        # pred_d = torch.exp(pred_log_d)
        return latent

    def get_latent_v2(self, input, targets): # Removed cuda() call
        latent = self.netG.forward(input, targets, False, True)
        # pred_log_d = prediction_d.squeeze(1)
        # pred_d = torch.exp(pred_log_d)
        return latent

    def L2(self, a, b):
        return torch.sum((a - b) ** 2)

    def latent_train(self, input_list, targets_list):
        latent1 = self.get_latent(input_list[0], targets_list[0])
        latent2 = self.get_latent(input_list[1], targets_list[1])

        print("Constraining between: ", targets_list[0]['img_1_path'][0].split(
            '/')[-2:], targets_list[1]['img_1_path'][0].split('/')[-2:])

        # just using L2 loss between the two latents for now
        loss = self.L2(latent1, latent2)
        # print("loss: ", loss)

        # taken from optimize_parameters
        #####
        self.optimizer_G.zero_grad()
        # self.backward_G(0) #TODO check n_iters
        loss.backward()
        self.optimizer_G.step()

    def depth_train(self, input, targets):
        # stack_inputs = autograd.Variable(input.cuda(), requires_grad=True)
        # prediction_d, pred_confidence = self.netG.forward(stack_inputs, targets)
        # print("got pred d", prediction_d.shape)
        # print("depth shape", depth.size())
        # reshaped_pred = torch.reshape(prediction_d, (1, 288, 512))

        # loss = self.L2(reshaped_pred, depth)
        # print(loss)
        # ###########
        # #optimize
        # ##########
        # self.optimizer_G.zero_grad()
        # # self.backward_G(0) #TODO check n_iters
        # loss.backward()
        # self.optimizer_G.step()

        self.input_images = autograd.Variable(input.cuda(), requires_grad=False)

        self.prediction_d, self.pred_confidence = self.netG.forward(
            self.input_images, targets)
        self.prediction_d = self.prediction_d.squeeze(1)

        self.targets = targets

        self.optimizer_G.zero_grad()
        self.backward_G(1)

    # Implementation of depth_train that increases effective batch size
    # If we use batch size 4, but desired effective batch size is 16, then
    # k = 16/4, and gradients are accumuluted before updating
    def depth_train(self, i, input, targets, number_batches):
        self.input_images = autograd.Variable(input.cuda(), requires_grad=False)

        self.prediction_d, self.pred_confidence = self.netG.forward(
            self.input_images, targets)
        self.prediction_d = self.prediction_d.squeeze(1)

        self.targets = targets

        self.backward_G(1)

        k = 4
        if ( (i+1) % k == 0 or (i+1) == number_batches):
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()

    # TODO: test with bigger batch
    def compute_latent_loss(self, input, targets):
        count = 0 # make sure to count only frames that have a valid next path
        total_loss = 0

        # print(img.shape) # [4, 3, 288, 512]
        # print(target['next_frame'].shape) # [4, 3, 288, 512]

        for i in range(input.shape[0]):
            # Check if valid next frame
            if (targets['next_frame'][i][0][0][0] != -1):
                count += 1
                #Get Latents
                latent_1 = self.get_latent_v2(input, targets)
                input_2 = targets["next_frame"][i]
                input_2 = torch.unsqueeze(input_2, 0)
                latent_2 = self.get_latent(input_2, targets)
                total_loss += self.L2(latent_1, latent_2)

        if (count == 0):
            return 0
        else:
            return total_loss/count

    def backward_G_latent(self, n_iter, input, targets):
        # Combined loss
        self.loss_joint = self.criterion_joint(self.input_images, self.prediction_d, None, self.targets)
        print('Supervision Train loss is %f ' % self.loss_joint)

        # add to tensorboard
        if n_iter % 100 == 0:
            self.write_summary('Train', self.input_images, self.prediction_d,
                               self.pred_confidence, self.targets, n_iter,
                               self.loss_joint)

        self.loss_joint_var = self.criterion_joint.get_loss_var()

        latent_loss = self.compute_latent_loss(input, targets)
        print('Latent Train loss is ' + str(self.latent_constraint_weight * latent_loss))

        combined_loss = self.loss_joint_var + self.latent_constraint_weight * latent_loss
        combined_loss.backward()
        # print(type(self.loss_joint_var), self.loss_joint_var.size())
        # self.loss_joint_var.backward()

    def depth_and_latent_train_v2(self, i, input, targets, number_batches):
        self.input_images = autograd.Variable(input.cuda(), requires_grad=True)

        self.prediction_d, self.pred_confidence = self.netG.forward(
            self.input_images, targets)
        self.prediction_d = self.prediction_d.squeeze(1)

        self.targets = targets

        # self.optimizer_G.zero_grad()
        self.backward_G_latent(1, input, targets)

        k = 1
        if ( (i+1) % k == 0 or (i+1) == number_batches):
            self.optimizer_G.step()
            self.optimizer_G.zero_grad()


    def depth_and_latent_train(self, input_list, targets_list):
        latent1 = self.get_latent(input_list[0], targets_list[0])
        latent2 = self.get_latent(input_list[1], targets_list[1])

        print("Constraining between: ", targets_list[0]['img_1_path'][0].split(
            '/')[-2:], targets_list[1]['img_1_path'][0].split('/')[-2:])

        # just using L2 loss between the two latents for now
        loss = self.L2(latent1, latent2)
        # print("loss: ", loss)

        # taken from optimize_parameters
        #####
        self.optimizer_G.zero_grad()
        # self.backward_G(0) #TODO check n_iters
        loss.backward()
        self.optimizer_G.step()

    def run_and_save_DAVIS(self, input_, targets, save_path, visualize):
        assert (self.num_input == 3)
        input_imgs = autograd.Variable(input_.cuda(), requires_grad=False)

        stack_inputs = input_imgs

        prediction_d, pred_confidence = self.netG.forward(
            stack_inputs, targets, visualize)
        pred_log_d = prediction_d.squeeze(1)
        pred_d = torch.exp(pred_log_d)
        # pred_d = pred_log_d

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for i in range(0, len(targets['img_1_path'])):

            youtube_dir = save_path + \
                targets['img_1_path'][i].split(
                    '/')[-2] + '/' + str(self.weights) + '/'

            if not os.path.exists(youtube_dir):
                os.makedirs(youtube_dir)

            saved_img = np.transpose(
                input_imgs[i, :, :, :].cpu().numpy(), (1, 2, 0))

            pred_d_ref = pred_d.data[i, :, :].cpu().numpy()

            output_path = youtube_dir + \
                targets['img_1_path'][i].split('/')[-1]
            print("Output path: ", output_path)
            # Added
            disparity = pred_d_ref
            disparity[disparity < 0] = 0
            # disparity = 1. / pred_d_ref
            disparity = disparity / np.max(disparity)
            disparity = np.tile(np.expand_dims(disparity, axis=-1), (1, 1, 3))
            saved_imgs = np.concatenate((saved_img, disparity), axis=1)
            saved_images = disparity
            saved_imgs = (saved_imgs*255).astype(np.uint8)

            imsave(output_path, saved_imgs)
    
        #hacky, for saving interim models while training
    def run_and_save_DAVIS_interim(self, input_, targets, save_path, visualize, weights):
        self.weights = weights
        self.run_and_save_DAVIS(input_, targets, save_path, visualize)
        return

    def switch_to_train(self):
        self.netG.train()

    def switch_to_eval(self):
        self.netG.eval()

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

    def update_learning_rate(self):
        self.scheduler.step()
        lr = self.optimizer_G.param_groups[0]['lr']
        print('Current learning rate = %.7f' % lr)
