from random import gauss
import torch, torchvision
import torch.nn as nn
from torchvision.utils import save_image
from resample import UpSample2d
from filter import LowPassFilter2d
import numpy as np

img = torchvision.io.read_image("test_img/small_baker.png").cuda() #, mode=torchvision.io.ImageReadMode.GRAY
upsample = UpSample2d(ratio=2).cuda()
# lowpass = LowPassFilter2d()
# original_paper_upsample = nn.UpsamplingBilinear2d(scale_factor=2)

img_reshaped = torch.unsqueeze(img, 0)
# print(img.shape)
# print(img_reshaped.shape)
upsampled_image = upsample(img) #first, upsample [[in spatial domain]] using the filter (but without the low pass!)

save_image(upsampled_image/255, "test_img/upsampled_baker_no_low_pass.png")

# fft_image = torch.fft.fft2(img/255., dim = (1,2))

