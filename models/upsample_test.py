import torch, torchvision
import torch.nn as nn
from torchvision.utils import save_image
from resample import UpSample2d

img = torchvision.io.read_image("/home/adam/Desktop/repos/small_baker.png", mode=torchvision.io.ImageReadMode.RGB).cuda()

# a = torch.ones((20,20,3)).cuda()
upsample = UpSample2d(ratio=2).cuda()
# original_upsample = nn.UpsamplingBilinear2d(scale_factor=2)

img_reshaped = torch.unsqueeze(img, 0)
# print(img.shape)
# print(img_reshaped.shape)
upsampled_image = upsample(img)
# original_upsampled_image = original_upsample(img_reshaped)
# save_image(upsampled_image/255, "baker_anti_alias_upsampled.png")
# save_image(original_upsampled_image/255, "baker_original_upsampled.png")

fft_image = torch.fft.fft2(img/255.)

fft_real = fft_image.real
fft_imaginary = fft_image.imag
# print(fft_imaginary.shape)
# print(fft_imaginary[0])

fft_real = upsample(fft_real)
fft_imaginary = upsample(fft_imaginary)

# print(fft_image[0])
resample_antialias= torch.fft.ifft2(torch.complex(fft_real, fft_imaginary)).real.clamp(0, 1)
# resample_antialias = torch.fft.ifft2(upsample(fft_image)).real.clamp(0, 1)
print(resample_antialias.shape)
# print(proper_resample.shape)
# print(proper_resample[0])
save_image(resample_antialias, "baker_fft_upsample.png")

# original_image = torch.fft.irfft2(fft_image)
# print(original_image.shape)
# save_image(original_image, "baker_fft_ifft.png")
