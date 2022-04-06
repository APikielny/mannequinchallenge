from random import gauss
import torch, torchvision
import torch.nn as nn
from torchvision.utils import save_image
from resample import UpSample2d
from filter import LowPassFilter2d
import numpy as np

img = torchvision.io.read_image("test_img/small_baker.png").cuda() #, mode=torchvision.io.ImageReadMode.GRAY

fft_image = torch.fft.fft2(img/255., dim = (1,2))

#### low pass filter:
x,y = torch.meshgrid(torch.arange(-250,250), torch.arange(-250,250))

circle_radius = 500
circle  = torch.fft.ifftshift(x*x + y*y < circle_radius^2)
# circle  = x*x + y*y < circle_radius^2
#############################

#### gaussian:
# sigma = 10
# # First a 1-D  Gaussian
# t = np.linspace(-10, 10, sigma)
# bump = np.exp(-0.1*t**2)
# bump /= np.trapz(bump) # normalize the integral to 1

# # make a 2-D kernel out of it
# kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
dst = np.sqrt(x*x+y*y)
 
# Initializing sigma and muu
sigma = 2
muu = 0.000
 
# Calculating Gaussian array
kernel = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
kernel /= np.sum(kernel)

print(kernel.shape)


#need both the padding and the fftshift for this to work, apparently. but also if you remove both, the blur looks better but it's not centered
# kernel = np.pad(kernel, int((img.shape[1] - sigma)/2 - 1))
# # shifted_fft_kernel = torch.fft.fft2(torch.fft.fftshift(torch.Tensor(kernel).cuda()), s = (500,500))
# shifted_fft_kernel = torch.fft.fft2(torch.fft.fftshift(torch.Tensor(kernel).cuda()), s = (500,500))
shifted_fft_kernel = torch.fft.fft2(torch.Tensor(kernel).cuda(), s = (500,500))

blurred = fft_image * shifted_fft_kernel


# fft_img_filtered = fft_image * circle.cuda()
fft_img_filtered = blurred

resample_antialias= torch.fft.ifft2(fft_img_filtered, dim = (1, 2)).real.clamp(0, 1)
save_image(resample_antialias, "test_img/baker_fft_gaussian.png")
