## imports
import torch
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import cv2
##

#this function taken from alias-free code
def design_lowpass_filter(numtaps, cutoff, width, fs, radial):
    assert numtaps >= 1
    # Identity filter.
    if numtaps == 1:
        return None
    # Separable Kaiser low-pass filter.
    if not radial:
        f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
        return torch.as_tensor(f, dtype=torch.float32)

    # Radially symmetric jinc-based filter.
    x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
    r = np.hypot(*np.meshgrid(x, x))
    f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
    beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
    w = np.kaiser(numtaps, beta)
    f *= np.outer(w, w)
    f /= np.sum(f)
    return torch.as_tensor(f, dtype=torch.float32)


def create_kernel(img_width, radial=False):
    # Hyperparameters.
    conv_kernel         = 3        # Convolution kernel size. Ignored for final the ToRGB layer.
    filter_size         = 6 #was 6        # Low-pass filter size relative to the lower resolution when up/downsampling.
    lrelu_upsampling    = 2        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
    use_radial_filters  = False    # Use radially symmetric downsampling filter? Ignored for critically sampled layers.
    conv_clamp          = 256      # Clamp the output to [-X, +X], None = disable clamping.
    magnitude_ema_beta  = 0.999    # Decay rate for the moving average of input magnitudes.
    torgb = False #TODO important, this determines if we are upsampling or just blurring I think

    #s (sampling rate) is "width of canvas in pixels"
    s = img_width
    tmp_sampling_rate = max(s, s) * (1 if torgb else lrelu_upsampling) #a bit edited from their code

    up_factor = int(np.rint(tmp_sampling_rate / s))
    print("computed upsampling factor: ", up_factor)
    # assert s * up_factor == tmp_sampling_rate

    #setting up_taps, from on their code
    up_taps = filter_size * up_factor if up_factor > 1 else 1 #(not sure what this part means, seems like a conversion from fourier to rgb i think): and not self.is_torgb else 1

    #width
    #"transition band  half width is (√2 - 1) * (s/2)"
    sqrt_2 = 1.41421356237
    in_half_width = (sqrt_2 - 1) * (s/2)
    width=in_half_width*2

    #cutoff, defined as f_c (in paper and code)
    #paper says f_c is s/2 for critical sampling, then moves to s/2 - f_h for non critical sampling
    # cutoff = s / 2
    cutoff = s / 2 - in_half_width

    upsample_filter = design_lowpass_filter(up_taps,cutoff,width,tmp_sampling_rate,radial)
    return upsample_filter

###############
#testing
###############

#############
#1d kernel
#############

# k = create_kernel(100)
# print("final kernel shape: ", k.shape)
# print("final kernel: ", k)

# plt.plot(k)
# plt.savefig("test_img/kernel.png")

###############
#radially symmetric kernel
###############
# radial_k = create_kernel(500, True)
# print("final radial kernel shape: ", k.shape)
# print("final radial kernel: ", k)

# normalized_kernel_img = radial_k.numpy()
# normalized_kernel_img = normalized_kernel_img - np.min(normalized_kernel_img)
# normalized_kernel_img /= np.max(normalized_kernel_img)
# normalized_kernel_img *= 255

# print(np.min(normalized_kernel_img), np.max(normalized_kernel_img))
# cv2.imwrite("test_img/windowed_sinc_from_paper_radial_img_width_500.png", normalized_kernel_img)

###############
#comparing kernels of diff img_widths (conclusion: they are the same?)
##############
# radial_k_500 = create_kernel(500, True)
# radial_k_100 = create_kernel(100, True)
# print(radial_k_500 - radial_k_100)