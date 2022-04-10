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
    def __init__(self, use_1x1_conv):
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
            self.list.append(
                nn.Sequential(
                    #nn.AvgPool2d(2),
                    # DownSample2d(),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    # Old Upsampling Filter
                    # nn.UpsamplingBilinear2d(scale_factor=2)
                )
            )  # EEE

        # print("self.list size", len(self.list))
        # print("self.list", self.list)
        # this gives confidence that self.list actually contains Channels1()

        for layer in self.list:
            layer.register_forward_hook(hook_fn)

    def forward(self, x):
        upsample = UpSample2d(ratio=2).cuda()
        downsample = DownSample2d().cuda()
        return self.list[0](x)+upsample(self.list[1](downsample(x)))

class Channels2(nn.Module):
    def __init__(self, use_1x1_conv):
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
                    Channels1(use_1x1_conv),
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
            )  # EF
            self.list.append(
                nn.Sequential(
                    #nn.AvgPool2d(2),
                    # DownSample2d(),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    Channels1(use_1x1_conv),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]]),
                    # Old Upsampling Filter
                    # nn.UpsamplingBilinear2d(scale_factor=2)
                )
            )  # EE1EF

        # for layer in self.list:
        #     layer.register_forward_hook(hook_fn)

    def forward(self, x):
        upsample = UpSample2d(ratio=2).cuda()
        downsample = DownSample2d().cuda()
        return self.list[0](x)+upsample(self.list[1](downsample(x)))

class Channels3(nn.Module):
    def __init__(self, use_1x1_conv):
        super(Channels3, self).__init__()
        self.list = nn.ModuleList()
        
        if use_1x1_conv:
            self.list.append(
                nn.Sequential(
                    inception(256, [[64], [1, 64, 64], [1, 64, 64], [1, 64, 64]]),
                    inception(256, [[128], [1, 64, 128], [1, 64, 128], [1, 64, 128]]),
                    Channels2(use_1x1_conv),
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
            self.list.append(
                nn.Sequential(
                    #nn.AvgPool2d(2),
                    # DownSample2d(),
                    inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                    inception(128, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    Channels2(use_1x1_conv),
                    inception(256, [[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]]),
                    inception(256, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                    # Old Upsampling Filter
                    # nn.UpsamplingBilinear2d(scale_factor=2)
                )
            )  # BD2EG
            self.list.append(
                nn.Sequential(
                    inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                    inception(128, [[32], [3, 64, 32], [7, 64, 32], [11, 64, 32]])
                )
            )  # BC

        # for layer in self.list:
        #     layer.register_forward_hook(hook_fn)

    def forward(self, x):
        upsample = UpSample2d(ratio=2).cuda()
        downsample = DownSample2d().cuda()
        return upsample(self.list[0](downsample(x)))+self.list[1](x)

class Channels4(nn.Module):
    def __init__(self, use_1x1_conv):
        super(Channels4, self).__init__()
        self.list = nn.ModuleList()
        
        if use_1x1_conv:
            self.list.append(
            nn.Sequential(
                inception(256, [[64], [1, 64, 64], [1, 64, 64], [1, 64, 64]]),
                inception(256, [[64], [1, 64, 64], [1, 64, 64], [1, 64, 64]]),
                Channels3(use_1x1_conv),
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
            self.list.append(
                nn.Sequential(
                    #nn.AvgPool2d(2),
                    # DownSample2d(),
                    inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                    inception(128, [[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]]),
                    Channels3(use_1x1_conv),
                    inception(128, [[32], [3, 64, 32], [5, 64, 32], [7, 64, 32]]),
                    inception(128, [[16], [3, 32, 16], [7, 32, 16], [11, 32, 16]]),
                    # TODO: state_dict error if UpSample2d is here
                    # Old Upsampling filter
                    # nn.UpsamplingBilinear2d(scale_factor=2)
                )
            )  # BB3BA
            self.list.append(
                nn.Sequential(
                    inception(128, [[16], [3, 64, 16], [7, 64, 16], [11, 64, 16]])
                )
            )  # A

        # for layer in self.list:
        #     layer.register_forward_hook(hook_fn)

    def forward(self, x):
        upsample = UpSample2d(ratio=2).cuda()
        downsample = DownSample2d().cuda()
        return upsample(self.list[0](downsample(x)))+self.list[1](x)


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

    # NOTE: Mapping size and scale are hyperparams
    def __init__(self, num_input_channels, mapping_size=128, scale=6):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size

        # TODO: Hard coding B for now. Issue is I need to init it once during training
        # and save in model state dictionary for testing to keep fourier features consistent

        # self._B = torch.randn((num_input_channels, mapping_size)) * scale
        self._B = torch.empty((num_input_channels, mapping_size))
        self._B[0] = torch.tensor(
        [  7.1344,   2.3810,  -4.4248,  -2.1793,  -8.9122,   5.6092,   8.7790,
         -5.3323,   8.9035,   0.4993,   4.7831,   1.9928,   7.0832,  15.3558,
         -4.7319,   3.9101,   8.0591,  -7.9647,   4.8088, -10.4289,   3.4868,
          2.8129,  -9.3382,   6.0375,  -4.3455,   0.6762,   8.8416,  -0.2466,
         -8.6647,  -8.1772,   1.4501,   3.4336,  15.0932,   2.2427,  -6.1435,
         11.8947,   5.8285,   4.6068,   3.1076,  -5.3516,  -3.7432,  -2.5859,
          3.8913,   4.9581,   5.2971,  12.9320,   2.0550,  -5.0567,  -0.1552,
        -10.3624, -11.3817,  -5.2694,  -0.6809,   0.8187,  -1.0186,  -9.4397,
          6.5233,   0.2519,  13.0533,   6.6493,  -6.2694,   4.7392,  -0.5058,
          8.5897,  -0.3021,   4.3687,   2.7382,  -2.2902,   8.4866,   0.1344,
         -4.2868,  -5.9618,   0.7390,  -7.6869,   2.1705,   4.7212,  -5.0691,
         11.7849,   2.3420,  -0.6555,   0.7803,   4.5135,   7.8613,   2.9653,
          3.9139,   9.8412,  -1.7845,   0.3563,   3.8456,   4.6100,  11.4645,
          5.5138,  -0.2337, -15.9702,  -5.0698,  -0.4843,   5.8527,  -4.4104,
          3.9721,   1.0792,  -4.0349,  -1.0934,  -0.8136,  -3.3727,  -2.8679,
          8.8586,  -1.0903,  -1.6446,   0.9572,   1.0830,   2.6495,  13.7116,
          6.3336,  -5.7626, -13.7837,  -3.8853,  -1.2342,  -3.2307,   5.0002,
          2.8380,  -1.0010,  -8.1745,   5.2710,  -6.6467,  15.6487,  -0.2287,
         -3.8237,  -1.4558]
        )
        self._B[1] = torch.tensor(
        [ 2.5379e-01,  7.9462e+00, -1.0474e+01,  4.7881e+00,  9.2552e+00,
        -1.7041e+00,  3.9223e+00, -3.5987e+00,  7.2413e+00, -4.9436e+00,
        -9.1166e+00,  4.0843e+00,  6.8498e+00, -5.7215e+00,  1.7181e+00,
        -8.5488e+00,  4.3599e+00, -7.5831e+00,  9.1362e+00,  6.7196e-01,
         7.7394e+00, -2.5151e+00, -4.9584e+00, -9.3628e+00,  5.5736e-01,
         3.3595e+00,  1.3097e+00,  1.2475e+01, -4.3886e+00,  3.7568e+00,
        -1.8355e+00, -4.6944e+00,  2.4884e+00, -3.8637e+00,  1.0861e-02,
         5.0710e+00, -1.4888e+01, -1.6329e-01, -3.1886e+00,  3.3626e+00,
        -5.6462e-01,  1.8711e+01, -2.2502e+00, -1.1601e+01,  4.1358e+00,
        -8.1253e+00,  1.1565e+00, -1.8919e+00,  5.2807e+00,  2.5040e+00,
        -7.3404e+00,  1.4309e+00,  5.2195e-01, -1.8865e+00,  7.0302e+00,
        -9.8594e+00,  1.8611e+00, -6.9617e+00,  7.4277e+00, -2.7440e-01,
        -1.1646e+01, -4.6427e+00,  5.9978e+00,  4.6580e+00, -6.6474e-01,
         1.9762e+00, -2.2546e+00,  2.0989e+00, -2.3451e+00,  8.0093e+00,
         3.2252e+00, -1.4209e+01,  3.7078e+00, -2.2873e+00,  7.4849e+00,
         5.0271e+00,  1.2748e+01, -2.3600e+00, -6.5682e-01,  5.6338e+00,
         1.7179e+00, -2.5277e+00, -7.2697e+00, -1.9488e+00, -1.3816e+00,
        -1.1854e+01,  4.5395e+00,  8.9846e+00,  7.3681e+00,  1.5766e+00,
        -7.4513e-01,  3.7154e+00, -1.7477e+00, -3.4447e+00, -2.7223e+00,
         5.2200e+00,  2.9180e-01, -3.5681e-01,  1.1176e+00,  4.7734e+00,
        -5.3679e+00,  1.4807e+00, -6.3439e+00,  3.6503e+00, -3.6584e+00,
        -3.1726e+00,  2.7002e+00,  3.8203e+00,  3.9671e+00, -2.6008e+00,
         4.3888e+00, -6.4011e+00, -5.8482e-01, -3.8900e+00,  2.4130e+00,
         3.1897e+00,  1.2141e+00,  3.0148e+00,  1.7875e+00, -1.2273e+00,
        -2.5690e+00,  1.1419e+01,  1.5173e+00,  5.0838e+00, -3.6544e+00,
         3.6841e+00,  8.7541e+00,  8.7311e-01]
        )
        self._B[2] = torch.tensor(
        [  0.0443,   3.9778,   0.9655,   4.8200,  -1.4327,  -6.6086,   3.3201,
         -4.5046,  -7.5334, -11.6904,   2.8458,   7.0414,  -8.4622,   2.1027,
         -3.0107,  -2.8342,  -3.8424,  -1.5130,   3.4110,  -4.9066,  -2.4134,
         10.4462,   1.4402,   2.4020,  10.4616,   2.8774,   3.6269,   3.7184,
         -3.0142,  -7.5353, -10.8281,   4.6023, -12.8916,   1.4613,   3.4577,
         -0.9991,   7.4374,  -0.5576, -10.4869, -13.7031,   8.5959,   3.2034,
         -8.7876,  10.0160,   1.0620,  -5.0062,   0.6169,   2.4225,  -0.5438,
          4.6321,  -9.8321,  12.4452,  -2.8816,  -5.5020,   3.8754,   9.7455,
         -7.7671,  -5.2154,  -5.5980,   0.0195,   2.3559,   1.6301,  -1.7598,
         -7.7880,  -9.4563,   0.8432,  11.7788,   1.1748,   3.7541,  -3.6137,
         -3.5381,   0.8068,  -5.6213,  16.2611,  -2.1631,  -1.7944,  -6.9886,
         -0.6734,   3.8014,  -4.7871,  -8.1848,  -5.2157,   3.5243,   6.5065,
          1.0242,   8.6966,   3.8656,   9.7577,   0.1342,  -1.6808,   9.1191,
          1.7602,   7.8891,  13.6552,  -1.4267,   1.4999,   7.8205,  -1.4839,
         12.4151,  -1.3242,  -7.2581,   1.0952,   8.1133,   1.3289,   4.0038,
          0.6330,  10.3591,   3.7559,   2.4339,  -3.5995,  -6.8406,  -3.8246,
         -5.4558,  -4.2637,   3.1334,   0.2224,  -5.2670,  -2.7876,  -1.2441,
          5.4898,  -2.3546,   4.4482,  11.7038,   5.4369,   6.7678,  -2.4052,
         -0.1400,  -3.1737]
        )
        self._B[3] = torch.tensor(
        [ -3.0544,   1.0770,   6.7171,  -2.8045,   4.6144,  -2.5249,   1.9325,
          1.8906,  -6.4072,   2.3998,   2.4674, -10.2307,  -0.9698,  -9.0180,
         -3.0033,   8.5845,  -2.1443,  -1.4908,   7.3861,   2.4307,   4.1968,
          0.0587,  -4.5107,  11.8293,   9.6016,  -2.2954,   4.8197,   7.5161,
          1.2726,   1.4668,  -0.3818,   0.3171,  -4.6502,  -4.2794,   4.3939,
         -2.6467,  -5.3514,   1.6539,   4.6510,  10.5594,  -3.8606,   3.8155,
          0.9761,   4.5416,  -1.5402,   3.5482,   1.9635,   2.0166,  -0.2418,
          3.6608,  -1.6901,   1.9288,  -2.8605,   2.7621,  -5.5373,   3.0389,
         -1.4942,  12.9436,   6.3484,  -4.4448,   0.3645, -10.9581,  -5.8416,
          1.4231,   5.4713,   1.7428,   4.1396,  12.8158,  -0.3059, -13.9547,
         -0.9726,  -1.4106,  -3.3790,  -3.6666,   0.6284,   6.9796,   1.3745,
          1.8375,  11.2592,  -4.3208,  -5.8846,  11.7159,  -8.5647,  -4.6677,
          8.2544,  -7.3628,  -5.9530,  -2.7731,  -3.4834,  -5.5197,   8.7223,
          9.8868,  -6.4917,   4.3090,  -1.4986,   7.0676,  -3.1735,   0.9612,
          4.3365,   2.2581,  -5.7226,  10.3018,  -2.2463,  -8.3335,  -6.9828,
        -16.2705, -17.2061,  -0.5334,   5.2238,   5.5437,  -9.7761,   4.4770,
         12.0056,   1.8479,  -3.9791,  -5.7744, -10.8246,  -3.9632,  -0.4967,
         18.4773,   0.8637,  -1.4778,  11.1706,   3.1975,   5.5608,  -7.8000,
         -2.6232,  -6.7131]
        )
        self._B[4] = torch.tensor(
        [ 11.1377,  -0.1715,  -9.4050,  -1.8583,   0.4806,  -1.1584,   8.9301,
          8.5339,  -0.5461,   4.7709,   5.3128,  -6.3135,  -8.7318,   3.3223,
          0.1505,   2.0846,   8.0877,  -3.7458,  -4.4726,   1.0991,  -1.7795,
         -7.9830,   6.3811,   3.1409,  -1.9476,   1.0898,  11.7748,   9.4228,
          8.6921,   0.2808,  -5.3947,   3.7735,  -0.6699,  -2.5962,   1.3704,
          2.8114,  -9.2087,  -8.3849,  -0.4334,  -7.9745, -10.1718,  -2.4641,
         -5.4770,  -1.3061,  -7.5370,  -7.1316,  -2.8031,   6.3314,   0.4897,
          5.9461,   4.5983,  -6.7346,   2.9780,   0.3408,  -8.8006, -12.4735,
          5.9435,  -2.5684,  -0.3271,   1.4360,   2.4096,  -2.9972,  -0.3749,
          9.9970,   7.4308,  -2.5513,   2.7988,   4.0151,   2.6166,   1.6104,
          1.4377, -11.9018,  -7.6141,   5.1960,   1.3513, -10.3130,   5.4915,
        -15.6765,   2.5907,  -5.3790,  -1.3759,  -8.1504, -12.5328,  -5.9271,
         -0.3600,   9.5912,   2.6056,  -1.3890,  -1.7023,   4.0153, -11.4221,
         -1.9905,   3.4844,  -4.8546,   6.2119, -11.7122,  -4.5354,  -2.1880,
         -4.5040,   3.5216,  -1.9541,   9.9974,   0.5736,  -3.6314, -12.9511,
          0.4163,   2.6579,   2.9084,   5.6800,  -3.0434,  -2.7534,   9.5868,
         -2.9490,  -0.1236,   7.5330,  10.6207, -10.3909,   4.4574,   2.4391,
          0.2270,   2.0330,  -1.6279,  11.3104,  -3.4861,  -2.7618,   7.7993,
         -4.3145,   1.3771]
        )

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
    def __init__(self, num_input, use_1x1_conv):
        super(HourglassModel, self).__init__()

        mapping_size = 128
        self.fourier_feature_transform = GaussianFourierFeatureTransform(5)

        # TODO: Switching to 1x1 Convolutions Works Currently, Just Testing
        # - I double feature maps for inception base 1x1 layer, which maybe isn't needed?
        # - Using inception with only 1x1 convolution may be unnecessary?
        # - Parameters overall decreased so can maybe switch mapping size to 256
        # TODO: (Maybe) Switch from doubling feature maps to something smarter especially for >3 kernel size 
        if use_1x1_conv:
            self.seq = nn.Sequential(
                nn.Conv2d(mapping_size * 2, 256, 1, padding = 0), 
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                Channels4(use_1x1_conv), 
            )

            uncertainty_layer = [
                nn.Conv2d(128, 1, 1, padding=0), torch.nn.Sigmoid()]
            self.uncertainty_layer = torch.nn.Sequential(*uncertainty_layer)
            self.pred_layer = nn.Conv2d(128, 1, 1, padding=0)
        else:
            self.seq = nn.Sequential(
                # nn.Conv2d(num_input, 128, 7, padding=3),
                # nn.Conv2d(5, 128, 7, padding = 3), # For r,g,b,x,y input
                nn.Conv2d(mapping_size * 2, 128, 7, padding = 3), 
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                Channels4(use_1x1_conv),
            )

            uncertainty_layer = [
                nn.Conv2d(64, 1, 3, padding=1), torch.nn.Sigmoid()]
            self.uncertainty_layer = torch.nn.Sequential(*uncertainty_layer)
            self.pred_layer = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, input_, targets, boolVisualize = False, latentOutput = False):

        split = targets['img_1_path'][0].split('/')
        frameName = split[-1][:-4]
        videoType = split[-2]

        # X,Y Grid Concat Logic Moved to Data Loader
        ff_input = self.fourier_feature_transform(input_)
        pred_feature = self.seq(ff_input)

        pred_d = self.pred_layer(pred_feature)
        pred_confidence = self.uncertainty_layer(pred_feature)

        if (boolVisualize):
            visualize(visualisation_feature_map, input_, videoType, frameName)

        if (latentOutput):
            latent = list(visualisation_feature_map.values())[1][0,:,:,:]
            return latent

        return pred_d, pred_confidence
