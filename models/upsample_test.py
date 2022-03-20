from random import gauss
import torch, torchvision
import torch.nn as nn
from torchvision.utils import save_image
from resample import UpSample2d
from filter import LowPassFilter2d
import numpy as np

img = torchvision.io.read_image("test_img/small_baker.png").cuda() #, mode=torchvision.io.ImageReadMode.GRAY
upsample = UpSample2d(ratio=2).cuda()
lowpass = LowPassFilter2d()
# original_paper_upsample = nn.UpsamplingBilinear2d(scale_factor=2)

img_reshaped = torch.unsqueeze(img, 0)
# print(img.shape)
# print(img_reshaped.shape)
upsampled_image = upsample(img)
# original_upsampled_image = original_paper_upsample(img_reshaped)
save_image(upsampled_image/255, "baker_anti_alias_upsampled.png")
# save_image(original_upsampled_image/255, "baker_original_upsampled.png")

fft_image = torch.fft.fft2(img/255., dim = (1,2))

fft_real = fft_image.real
fft_imaginary = fft_image.imag
# print("fft imaginary", fft_imaginary.shape)
# print(fft_real.shape)
# print(fft_imaginary[0])

# fft_real = upsample(fft_real)
# fft_imaginary = upsample(fft_imaginary)



# https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_image_blur.html
#######################################
sigma = 20
# First a 1-D  Gaussian
t = np.linspace(-10, 10, sigma)
bump = np.exp(-0.1*t**2)
bump /= np.trapz(bump) # normalize the integral to 1

# make a 2-D kernel out of it
kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

kernel = np.pad(kernel, int((img.shape[1] - sigma)/2 - 1))

print("padded kernel shape", kernel.shape)

save_image(torch.Tensor(kernel)*255, "test_img/gaussian_kernel_sigma_{}_pre_fft.png".format(str(sigma)))


# Padded fourier transform, with the same shape as the image
# We use :func:`scipy.signal.fftpack.fft2` to have a 2D FFT
kernel_ft = torch.fft.fftshift(torch.fft.fft2(torch.Tensor(kernel).cuda(), s = (500,500), dim = (0,1))) #, shape=img.shape[:2], axes=(0, 1))
kernel_ft_no_pad = torch.fft.fft2(torch.Tensor(kernel).cuda())

print("shape fft img: ", fft_real.shape)

# kernel_ft_pad_real = torch.zeros((fft_real.shape))
# kernel_ft_pad_real[:30] = kernel_ft.real
# kernel_ft_pad_imag = torch.zeros((fft_imaginary.shape))
# kernel_ft_pad_imag[:30] = kernel_ft.imag
# kernel_ft_pad = torch.complex(kernel_ft_pad_real, kernel_ft_pad_real)

# convolve
# img_ft = torch.fft.fft2(img, axes=(0, 1))
# # the 'newaxis' is to match to color direction
# img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
# img2 = torch.fft.ifft2(img2_ft, axes=(0, 1)).real

fft_real = fft_real * kernel_ft.real
fft_imaginary = fft_imaginary * kernel_ft.imag


# clip values to range
# img2 = np.clip(img2, 0, 1)
######################################

kernel_out = torch.fft.ifft2(torch.fft.ifftshift(kernel_ft)).real.clamp(0,1)
save_image(kernel_out*255, "test_img/gaussian_kernel_sigma_{}.png".format(str(sigma)))

# save_image(torch.fft.fft2(img/255., dim = (1,2)).real, "test_img/real_fft_of_image.png")
# save_image(torch.fft.ifft2(torch.fft.fft2(img/255., dim = (1,2))).real.clamp(0, 1), "test_img/real_ifft_fft_of_image.png")
# save_image(torch.fft.ifft2(torch.fft.ifftshift(torch.fft.fftshift(torch.fft.fft2(img/255., dim = (1,2))))).real.clamp(0, 1), "test_img/real_ifft_fft_of_image_with_shifts.png")



# # print(fft_image[0])
resample_antialias= torch.fft.ifft2(torch.complex(fft_real, fft_imaginary), dim = (1, 2)).real.clamp(0, 1)
# # resample_antialias = torch.fft.ifft2(upsample(fft_image)).real.clamp(0, 1)
# # print(resample_antialias.shape)
# # print(proper_resample.shape)
# # print(proper_resample[0])
save_image(resample_antialias, "test_img/baker_fft_blah.png")

# # original_image = torch.fft.irfft2(fft_image)
# # print(original_image.shape)
# # save_image(original_image, "baker_fft_ifft.png")
