from pickle import TRUE
from random import gauss
import torch, torchvision
import torch.nn as nn
from torchvision.utils import save_image
from resample import UpSample2d
from filter import LowPassFilter2d
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2

from sinc_filter import create_kernel

def gaussian_filter_fn(img):
    return torch.Tensor(gaussian_filter(img.cpu(), sigma=sigma_var))

def sinc_filter_spatial(img):
    #https://stackoverflow.com/questions/68930662/two-dimensional-sinc-function
    # equivalently (thanks @Kelly Bundy):
    x1 = np.linspace(-10, 10, 50)
    x2 = np.linspace(-10, 10, 50)
    sinc2d = np.outer(np.sin(x1), np.sin(x2)) / np.outer(x1, x2)
    print(sinc2d.shape)
    save_image(torch.Tensor(sinc2d), "test_img/sinc.png")
    spatial_sinc = torch.fft.ifftshift(torch.fft.ifft2(torch.Tensor(sinc2d))).real
    save_image(torch.Tensor(spatial_sinc)*255, "test_img/ifft_sinc_fftshift_after.png")
    print("spatial sinc: ", spatial_sinc)

    # increased_dims_sinc = np.repeat(spatial_sinc[np.newaxis, :, :], 3, axis=0)
    # increased_dims_sinc = np.repeat(spatial_sinc[np.newaxis, :, :, :], 3, axis=0) #TODO is this correct?
    # print("increased dims dims", increased_dims_sinc.shape)

    # kernel = torch.nn.Conv2d(3, 3, 50, stride=1)
    # kernel.weight = torch.nn.Parameter(increased_dims_sinc, requires_grad = False)

    # out_img = kernel(img)

    img_numpy = img.cpu().numpy()[:3, :, :].reshape((1000, 1000, 3))
    kernel_numpy = spatial_sinc.cpu().numpy()
    print(img_numpy.shape)
    print(kernel_numpy.shape)
    out_img = cv2.filter2D(src=img_numpy, ddepth=-1, kernel=kernel_numpy)
    return torch.Tensor(torch.reshape(torch.Tensor(out_img), (3,1000,1000)))

def get_gaussian_kernel():
    x, y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
    dst = np.sqrt(x*x+y*y)
    
    # Initializing sigma and muu
    sigma = 1
    muu = 0.000
 
    # Calculating Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    gauss = gauss/np.sum(gauss)

    return gauss

def windowed_sinc_from_alias_free(img, sampling_factor):
    use_radial = False

    # img_numpy = img.cpu().numpy()
    img_numpy = img.cpu().numpy()[:3, :, :].reshape((250, 250, 3))
    width = img.shape[2] * (1.0/sampling_factor)
    print("width:", width)
    windowed_sinc_kernel = create_kernel(width, use_radial).cpu().numpy()

    ##############################################
    #hard code
    # windowed_sinc_kernel = np.array([0, 0.5, 0.5])
    #try with gauss for sanity:
    # windowed_sinc_kernel = get_gaussian_kernel()
    # use_radial = TRUE
    # windowed_sinc_kernel = np.array([[0.111, 0.111, 0.111], [0.111, 0.111, 0.111], [0.111, 0.111, 0.111]])
    ##############################################

    print("sum of kernel weights", np.sum(windowed_sinc_kernel))
    print("kernel shape", windowed_sinc_kernel.shape)
    print("kernel weights", windowed_sinc_kernel)
    # windowed_sinc_kernel = np.array([1])
    
    if use_radial:
        out_img = cv2.filter2D(src=img_numpy, ddepth=-1, kernel=windowed_sinc_kernel, borderType=4)
    else:
        #for 1d kernel:
        # print("reshaped kernel: ", np.shape(np.reshape(windowed_sinc_kernel, (1, -1))))

        out_img = cv2.filter2D(src=img_numpy, ddepth=-1, kernel=windowed_sinc_kernel)
        out_img = cv2.filter2D(src=out_img, ddepth=-1, kernel=np.reshape(windowed_sinc_kernel, (1, -1)))


    # out_img = out_img[:,::2,:]

    return torch.Tensor(torch.reshape(torch.Tensor(out_img), (3,250,250)))

# def my_upsample(img, factor=2):
#     numpy_img = img.cpu().numpy()
#     # upsampled = np.repeat(numpy_img, 2, axis = 1)
#     upsampled = torch.zeros((img.shape()[0], img.shape()[1]*2, img.shape()[2]), device=x.device)
#     upsampled[..., ::self.ratio, ::self.ratio] = 0
#     return torch.Tensor(upsampled)

def upsample_testing(img):
    ###############
    ##### upsample
    #############

    upsample = UpSample2d(ratio=2).cuda()
    # original_paper_upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    # img_reshaped = torch.unsqueeze(img, 0)
    # print(img.shape)
    # print(img_reshaped.shape)
    upsampled_image = upsample(img) #first, upsample [[in spatial domain]] using the filter (but without the low pass!)
    ##############
    ##############

    # sigma_var = 3
    blurred = windowed_sinc_from_alias_free(upsampled_image, 2)

    # save_image(my_upsample(img)/255, "test_img/rewrite_upsample_function.png")

    # save_image(blurred/255, "test_img/upsampled_baker_img_windowed_sinc_blur_from_paper_testing_radial_with_f_c_equal_non_critical_sampling_and_s_old_width_filter_size_1.png")
    save_image(blurred/255, "test_img/upsampled_baker_img_windowed_sinc_hard_code_test.png")
    # save_image(blurred/255, "test_img/no_upsample_just_windowed_sinc_blur_from_paper_testing_torgb_True.png")
    # fft_image = torch.fft.fft2(img/255., dim = (1,2))
    return

def my_downsample(img, factor=2):
    return img[:,::factor,::factor]

def downsample_testing(img):
    downsampled = my_downsample(img)
    # blurred = windowed_sinc_from_alias_free(downsampled, 1/2)
    import scipy.ndimage
    # kernel = get_gaussian_kernel()
    kernel = create_kernel(100*2, True).cpu().numpy()
    kernel = np.reshape(kernel, (1, 12, 12))
    blurred = torch.Tensor(scipy.ndimage.convolve(downsampled.cpu().numpy(), kernel))
    save_image(blurred/255, "test_img/downsampling/downsampled_baker_with_sinc_sinc_filter_scipy_check.png")
    return

#bw working version
# def downsample_testing_torch(img):
#     downsampled = my_downsample(img).cpu()[0, :, :].float()
#     downsampled = torch.reshape(downsampled, (1, 1, 250, 250))
#     # print("bw shape: ", downsampled.shape)
#     # blurred = windowed_sinc_from_alias_free(downsampled, 1/2)
#     # kernel = get_gaussian_kernel()
#     kernel = create_kernel(100*2, True).cpu().numpy()
#     kernel = torch.Tensor(np.reshape(kernel, (1, 1, 12, 12)))
#     # kernel = torch.Tensor([0,0,0,0,1,0,0,0,0]).float()
#     kernel = torch.reshape(kernel, (1,1,12,12))
#     # kernel = kernel.repeat(1, 3, 1, 1)
#     print("repeated kernel shape, ", kernel.shape)

#     print("kernel type", type(kernel))
#     print("image type", type(downsampled))


#     conv = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=(1, 1, 12, 12), stride=1, bias=False)
#     with torch.no_grad():
#         conv.weight = torch.nn.Parameter(kernel)
#         blurred = conv(downsampled)

#     save_image(blurred/255, "test_img/downsampling/downsampled_baker_with_sinc_sinc_filter_torch_attempt.png")


def downsample_testing_torch(img):
    downsampled = my_downsample(img).cpu().float()
    # downsampled = torch.reshape(downsampled, (1, 4, 250, 250))
    # print("bw shape: ", downsampled.shape)
    # blurred = windowed_sinc_from_alias_free(downsampled, 1/2)
    # kernel = get_gaussian_kernel()
    kernel = create_kernel(100*2, True).cpu().numpy()
    kernel = torch.Tensor(np.reshape(kernel, (1, 1, 12, 12)))
    # kernel = torch.Tensor([0,0,0,0,1,0,0,0,0]).float()
    kernel = torch.reshape(kernel, (1,1,12,12))
    # kernel = kernel.repeat(1, 3, 1, 1)
    print("repeated kernel shape, ", kernel.shape)

    print("kernel type", type(kernel))
    print("image type", type(downsampled))


    conv = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=(1, 1, 12, 12), stride=1, bias=False)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(kernel)
        blurred1 = conv(torch.reshape(downsampled[0, :, :], (1, 1, 250, 250)))
        blurred2 = conv(torch.reshape(downsampled[1, :, :], (1, 1, 250, 250)))
        blurred3 = conv(torch.reshape(downsampled[2, :, :], (1, 1, 250, 250)))
    blurred_combine = torch.stack([blurred1, blurred2, blurred3])
    print("shape combination", blurred_combine.shape)
    blurred_combine = torch.reshape(blurred_combine, (1, 3, 239, 239))
    print("shape combination post shift", blurred_combine.shape)

    save_image(blurred_combine/255, "test_img/downsampling/downsampled_baker_with_sinc_sinc_filter_torch_attempt.png")

def upsample_testing_torch(img):
    upsample = UpSample2d(ratio=2).cuda()
    upsampled_image = upsample(img).cpu().float()

    kernel = create_kernel(100/2, True).cpu().numpy()
    kernel = torch.Tensor(np.reshape(kernel, (1, 1, 12, 12)))
    kernel = torch.reshape(kernel, (1,1,12,12))
    # kernel = kernel.repeat(1, 3, 1, 1)
    # print("repeated kernel shape, ", kernel.shape)

    # print("kernel type", type(kernel))
    # print("image type", type(downsampled))


    conv = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size=(1, 1, 12, 12), stride=1, bias=False)
    with torch.no_grad():
        conv.weight = torch.nn.Parameter(kernel)
        blurred1 = conv(torch.reshape(upsampled_image[0, :, :], (1, 1, 1000, 1000)))
        blurred2 = conv(torch.reshape(upsampled_image[1, :, :], (1, 1, 1000, 1000)))
        blurred3 = conv(torch.reshape(upsampled_image[2, :, :], (1, 1, 1000, 1000)))
    blurred_combine = torch.stack([blurred1, blurred2, blurred3])
    print("shape combination", blurred_combine.shape)
    blurred_combine = torch.reshape(blurred_combine, (1, 3, 989, 989))
    print("shape combination post shift", blurred_combine.shape)

    save_image(blurred_combine/255, "test_img/upsampled_baker_with_sinc_sinc_filter_torch_attempt.png")



img = torchvision.io.read_image("test_img/small_baker.png").cuda() #, mode=torchvision.io.ImageReadMode.GRAY

# upsample_testing(img)
# downsample_testing_torch(img)
upsample_testing_torch(img)