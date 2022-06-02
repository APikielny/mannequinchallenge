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
import torch.nn as nn
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from .resample_v2 import DownSample2d, UpSample2d

from visualize import visualize_layer, visualize, view_all_activation_maps


class inception(nn.Module):
    def __init__(self, input_size, config):
        # Sample input: 128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]
        # 128 - Number of input channels
        # [32] - Number of layers for base 1*1 conv layer
        # [3, 32, 32] - kernel size = 3, out_a = 32, out_b = 3

        self.config = config
        super(inception, self).__init__()
        self.convs = nn.ModuleList()

        # Base 1*1 conv layer
        self.convs.append(nn.Sequential(
            nn.Conv2d(input_size, config[0][0], 1),
            nn.BatchNorm2d(config[0][0], affine=False),
            nn.ReLU(True),
        ))

        # Additional layers
        for i in range(1, len(config)):
            filt = config[i][0]
            pad = int((filt-1)/2)
            out_a = config[i][1]
            out_b = config[i][2]
            conv = nn.Sequential(
                nn.Conv2d(input_size, out_a, 1),
                nn.BatchNorm2d(out_a, affine=False),
                nn.ReLU(True),
                nn.Conv2d(out_a, out_b, filt, padding=pad),
                nn.BatchNorm2d(out_b, affine=False),
                nn.ReLU(True)
            )
            self.convs.append(conv)

    def __repr__(self):
        return "inception"+str(self.config)

    def forward(self, x):
        ret = []
        for conv in (self.convs):
            ret.append(conv(x))
        return torch.cat(ret, dim=1)

visualisation_feature_map = {}

#type is Sequential, torch.tensor
#callback from activation hook

def hook_fn(m, i, o):
    # print(m)
    visualisation_feature_map[m] = o

class Channels1(nn.Module):
    def __init__(self, use_1x1_conv, anti_alias_upsample=False, anti_alias_downsample=False):
        super(Channels1, self).__init__()

        self.list = nn.ModuleList()

        if use_1x1_conv:
            self.list.append(
                nn.Sequential(
                    inception(512, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]]),
                    inception(512, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]])
                )
            ) 
            self.list.append(
                nn.Sequential(
                    inception(512, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]]),
                    inception(512, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]]),
                    inception(512, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]]),
                )
            ) 
        else:
            self.list.append(
                nn.Sequential(
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
                )
            )  # EE

            layers = [inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])]
            if anti_alias_downsample:
                layers.insert(0, DownSample2d().cuda())
            else:
                layers.insert(0, nn.AvgPool2d(2))

            if anti_alias_upsample:
                layers.append(UpSample2d(ratio=2).cuda())
            else:
                layers.append(nn.UpsamplingBilinear2d(scale_factor=2))

            self.list.append(
                nn.Sequential(*layers)
            )  # EEE

        # print("self.list size", len(self.list))
        # print("self.list", self.list)
        # this gives confidence that self.list actually contains Channels1()

        for layer in self.list:
            layer.register_forward_hook(hook_fn)

    def forward(self, x):
        # upsample = UpSample2d(ratio=2).cuda()
        # downsample = DownSample2d().cuda()
        return self.list[0](x)+self.list[1](x)
        # return self.list[0](x)+upsample(self.list[1](downsample(x)))
        # return self.list[0](x)+self.list[1](downsample(x))

class Channels2(nn.Module):
    def __init__(self, use_1x1_conv, anti_alias_upsample=False, anti_alias_downsample=False):
        super(Channels2, self).__init__()
        self.list = nn.ModuleList()

        if use_1x1_conv:
            self.list.append(
                nn.Sequential(
                    inception(512, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]]),
                    inception(512, [[128], [1, 128, 128], [1, 128, 128], [1, 128, 128]])
                )
            )  
            self.list.append(
                nn.Sequential(
                    inception(512, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]]),
                    inception(512, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]]),
                    Channels1(use_1x1_conv, anti_alias_upsample, anti_alias_downsample),
                    inception(512, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]]),
                    inception(512, [[128], [1, 128, 128], [1, 128, 128], [1, 128, 128]]),
                )
            )
        else:
            self.list.append(
                nn.Sequential(
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]])
                )
            ) 
            # EF
            # self.list.append(
            #     nn.Sequential(
            #         #nn.AvgPool2d(2),
            #         # DownSample2d(),
            #         # inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
            #         # inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
            #         # Channels1(use_1x1_conv, anti_alias_upsample, anti_alias_downsample),
            #         # inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
            #         # inception(256, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]]),
            #         # Old Upsampling Filter
            #         # nn.UpsamplingBilinear2d(scale_factor=2)
            #     )
            # )  # EE1EF

            layers = [inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    Channels1(use_1x1_conv, anti_alias_upsample, anti_alias_downsample),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]])]

            if anti_alias_downsample:
                layers.insert(0, DownSample2d().cuda())
            else:
                layers.insert(0, nn.AvgPool2d(2))

            if anti_alias_upsample:
                layers.append(UpSample2d(ratio=2).cuda())
            else:
                layers.append(nn.UpsamplingBilinear2d(scale_factor=2))

            self.list.append(nn.Sequential(*layers))

        # for layer in self.list:
        #     layer.register_forward_hook(hook_fn)

    def forward(self, x):
        # upsample = UpSample2d(ratio=2).cuda()
        # downsample = DownSample2d().cuda()
        # return self.list[0](x)+self.list[1](downsample(x))
        # return self.list[0](x)+upsample(self.list[1](downsample(x)))
        return self.list[0](x)+self.list[1](x)


class Channels3(nn.Module):
    def __init__(self, use_1x1_conv, anti_alias_upsample=False, anti_alias_downsample=False):
        super(Channels3, self).__init__()
        self.list = nn.ModuleList()
        
        if use_1x1_conv:
            self.list.append(
                nn.Sequential(
                    inception(256, [[64], [1, 64, 64], [1, 64, 64], [1, 64, 64]]),
                    inception(256, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]]),
                    Channels2(use_1x1_conv, anti_alias_upsample, anti_alias_downsample),
                    inception(512, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]]),
                    inception(512, [[64], [1, 64, 64], [1, 64, 64], [1, 64, 64]]),
                )
            ) 
            self.list.append(
                nn.Sequential(
                    inception(256, [[64], [1, 64, 64], [1, 64, 64], [1, 64, 64]]),
                    inception(256, [[64], [1, 128, 64], [1, 128, 64], [1, 128, 64]])
                )
            )
        else:
            # self.list.append(
            #     nn.Sequential(
            #         #nn.AvgPool2d(2),
            #         # DownSample2d(),
            #         inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
            #         inception(128, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
            #         Channels2(use_1x1_conv, anti_alias_upsample, anti_alias_downsample),
            #         inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
            #         inception(256, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
            #         # Old Upsampling Filter
            #         # nn.UpsamplingBilinear2d(scale_factor=2)
            #     )
            # )  # BD2EG

            layers = [inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                    inception(128, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    Channels2(use_1x1_conv, anti_alias_upsample, anti_alias_downsample),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),]

            if anti_alias_downsample:
                layers.insert(0, DownSample2d().cuda())
            else:
                layers.insert(0, nn.AvgPool2d(2))

            if anti_alias_upsample:
                layers.append(UpSample2d(ratio=2).cuda())
            else:
                layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.list.append(nn.Sequential(*layers))

            self.list.append(
                nn.Sequential(
                    inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                    inception(128, [[32], [3, 64, 32], [7, 64, 32], [11, 64, 32]])
                )
            )  # BC

        # for layer in self.list:
        #     layer.register_forward_hook(hook_fn)

    def forward(self, x):
        # upsample = UpSample2d(ratio=2).cuda()
        # downsample = DownSample2d().cuda()
        return self.list[0](x)+self.list[1](x)
        # return upsample(self.list[0](downsample(x)))+self.list[1](x)
        # return self.list[0](downsample(x))+self.list[1](x)


class Channels4(nn.Module):
    def __init__(self, use_1x1_conv, anti_alias_upsample=False, anti_alias_downsample=False):
        super(Channels4, self).__init__()
        self.list = nn.ModuleList()
        
        if use_1x1_conv:
            self.list.append(
            nn.Sequential(
                inception(256, [[64], [1, 64, 64], [1, 64, 64], [1, 64, 64]]),
                inception(256, [[64], [1, 64, 64], [1, 64, 64], [1, 64, 64]]),
                Channels3(use_1x1_conv, anti_alias_upsample, anti_alias_downsample),
                inception(256, [[64], [1, 128, 64], [1, 128, 64], [1, 128, 64]]),
                inception(256, [[32], [1, 64, 32], [1, 64, 32], [1, 64, 32]]),
            )
            )  
            self.list.append(
                nn.Sequential(
                    inception(256, [[32], [1, 128, 32], [1, 128, 32], [1, 128, 32]])
                )
            )  
        else: 
            # self.list.append(
            #     nn.Sequential(
            #         #nn.AvgPool2d(2),
            #         # DownSample2d(),
            #         inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
            #         inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
            #         Channels3(use_1x1_conv, anti_alias_upsample, anti_alias_downsample),
            #         inception(128, [[32], [3, 64, 32], [5, 64, 32], [7, 64, 32]]),
            #         inception(128, [[16], [3, 32, 16], [7, 32, 16], [11, 32, 16]]),
            #         # Old Upsampling filter
            #         # nn.UpsamplingBilinear2d(scale_factor=2)
            #     )
            # )  # BB3BA

            layers = [inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                    inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                    Channels3(use_1x1_conv, anti_alias_upsample, anti_alias_downsample),
                    inception(128, [[32], [3, 64, 32], [5, 64, 32], [7, 64, 32]]),
                    inception(128, [[16], [3, 32, 16], [7, 32, 16], [11, 32, 16]])]

            if anti_alias_downsample:
                layers.insert(0, DownSample2d().cuda())
            else:
                layers.insert(0, nn.AvgPool2d(2))

            if anti_alias_upsample:
                layers.append(UpSample2d(ratio=2).cuda())
            else:
                layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
            
            self.list.append(nn.Sequential(*layers))

            self.list.append(
                nn.Sequential(
                    inception(128, [[16], [3, 64, 16], [7, 64, 16], [11, 64, 16]])
                )
            )  # A

        # for layer in self.list:
        #     layer.register_forward_hook(hook_fn)

    def forward(self, x):
        # upsample = UpSample2d(ratio=2).cuda()
        # downsample = DownSample2d().cuda()
        return self.list[0](x)+self.list[1](x)
        # return upsample(self.list[0](downsample(x)))+self.list[1](x)
        # return self.list[0](downsample(x))+self.list[1](x)


        # return self.list[0](x)+self.list[1](x)

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=128, scale=6):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size

        self._B = torch.randn((num_input_channels, mapping_size)) * scale
        self.register_parameter(name = "fourier-matrix", param = torch.nn.parameter.Parameter(self._B, requires_grad=False))

    def forward(self, x):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

class HourglassModel(nn.Module):
    def __init__(self, num_input, use_1x1_conv, scale, anti_alias_upsample, anti_alias_downsample, _isTrain, useFourier):
        super(HourglassModel, self).__init__()
        self.useFourier = useFourier
        print("Set useFourier variable.")
        print("Num input: ", num_input)

        if self.useFourier:
            # Hyperparameters for fourier features
            self.mapping_size = 128
            self.scale = scale
            self.fourier_feature_transform = GaussianFourierFeatureTransform(5, self.mapping_size, self.scale)

            # TODO: Switching to 1x1 Convolutions Works Currently, Just Testing
            # - I double feature maps for inception base 1x1 layer, which maybe isn't needed?
            # - Using inception with only 1x1 convolution may be unnecessary/weird? Research more about inception purpose
            # - Parameters overall decreased so can maybe switch mapping size to 256
            # Some more overfitting + learning rate type tests, feels weird to compare because parameter count is smaller by factor 2x
            # TODO: (Maybe) Switch from doubling feature maps to something smarter especially for >3 kernel size 
            #this should be specified with a fourier-features frequency.
            #using 1x1s without fourier features doesn't make sense because then there is no positional encoding
            if use_1x1_conv: 
                self.seq = nn.Sequential(
                    nn.Conv2d(self.mapping_size * 2, 256, 1, padding = 0), 
                    nn.BatchNorm2d(256),
                    nn.ReLU(True),
                    Channels4(use_1x1_conv, anti_alias_upsample, anti_alias_downsample), 
                )

                uncertainty_layer = [
                    nn.Conv2d(128, 1, 1, padding=0), torch.nn.Sigmoid()]
                self.uncertainty_layer = torch.nn.Sequential(*uncertainty_layer)
                self.pred_layer = nn.Conv2d(128, 1, 1, padding=0)
            else:
                self.seq = nn.Sequential(
                    # nn.Conv2d(num_input, 128, 7, padding=3),
                    # nn.Conv2d(5, 128, 7, padding = 3), # For r,g,b,x,y input
                    nn.Conv2d(self.mapping_size * 2, 128, 7, padding = 3), 
                    nn.BatchNorm2d(128),
                    nn.ReLU(True),
                    Channels4(use_1x1_conv, anti_alias_upsample, anti_alias_downsample),
                )

                uncertainty_layer = [
                    nn.Conv2d(64, 1, 3, padding=1), torch.nn.Sigmoid()]
                self.uncertainty_layer = torch.nn.Sequential(*uncertainty_layer)
                self.pred_layer = nn.Conv2d(64, 1, 3, padding=1)
        else:
            self.seq = nn.Sequential(
            nn.Conv2d(num_input, 128, 7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            Channels4(use_1x1_conv, anti_alias_upsample, anti_alias_downsample),
            )

            uncertainty_layer = [
                nn.Conv2d(64, 1, 3, padding=1), torch.nn.Sigmoid()]
            self.uncertainty_layer = torch.nn.Sequential(*uncertainty_layer)
            self.pred_layer = nn.Conv2d(64, 1, 3, padding=1)



    def forward(self, input_, targets, boolVisualize = False, latentOutput = False):
        print("hourglass forward pass")
        print("useFourier: ", self.useFourier)
        split = targets['img_1_path'][0].split('/')
        frameName = split[-1][:-4]
        videoType = split[-2]
        print("Split.")

        if self.useFourier:
            # X,Y Grid Concat Logic Moved to Data Loader
            ff_input = self.fourier_feature_transform(input_)
            pred_feature = self.seq(ff_input)
        else:
            print("Calling self.seq")
            print("Input size: ", input_.shape)
            pred_feature = self.seq(input_)

        print("hourglass forward pass, got pred feature")

        pred_d = self.pred_layer(pred_feature)
        pred_confidence = self.uncertainty_layer(pred_feature)

        if (boolVisualize):
            visualize(visualisation_feature_map, input_, videoType, frameName)

        if (latentOutput):
            latent = list(visualisation_feature_map.values())[1][0,:,:,:]
            return latent
        print("hourglass forward pass, returning")

        return pred_d, pred_confidence
