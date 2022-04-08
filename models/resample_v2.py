import torch
import torch.nn as nn
import torch.nn.functional as F
#from .filter import LowPassFilter1d, LowPassFilter2d
#from filter import LowPassFilter1d, LowPassFilter2d
from .sinc_filter import design_lowpass_filter, create_kernel

class LowPassWindowedSinc(nn.Module):
    def __init__(self,
                ratio=2,
                pad=True):
        super().__init__()
        
        self.pad = pad
    
    def forward(self, x):
        shape = list(x.shape)
        width = shape[-1]
        height = shape[-2]

        kernel = create_kernel(width, True).cuda() #create kernel based on dims of x
        #separable (non radial) version:
        # kernel = create_kernel(width, False).cuda() #create kernel based on dims of x
        # kernel = torch.outer(kernel, kernel)
        
        kernel = torch.unsqueeze(kernel, 0) #add dims to front to be (1,1,h,w)
        kernel = torch.unsqueeze(kernel, 0)
        kernel_size = kernel.shape[-1]
        self.half_size = kernel_size // 2
        self.filter = kernel

        x = x.view(-1, 1, shape[-2], shape[-1]) #this reshape is necessary for convolution to work
        
        if self.pad:
            x = F.pad(
                x, (self.half_size, self.half_size, self.half_size,
                    self.half_size),
                mode='constant',
                value=0)  # empirically, it is better than replicate or reflect
            #mode='replicate')
        
        out = F.conv2d(x, self.filter)
        new_shape = shape[:-2] + [height, width]
        cropped = out[:,:,:height,:width]
        
        return cropped.reshape(new_shape) #undo the reshape

class UpSample2d(nn.Module):
    def __init__(self, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.lowpass = LowPassWindowedSinc()

    def forward(self, x):
        shape = list(x.shape)
        # print(shape)
        new_shape = shape[:-2] + [shape[-2] * self.ratio
                                  ] + [shape[-1] * self.ratio]

        xx = torch.zeros(new_shape, device=x.device)
        #shape + [self.ratio**2], device=x.device)
        xx[..., ::self.ratio, ::self.ratio] = x
        xx = self.ratio**2 * xx
        x = self.lowpass(xx)

        # print("xx shape", xx.shape)
        return x

class DownSample2d(nn.Module):
    def __init__(self, ratio=2):
        super().__init__()
        self.ratio = ratio
        self.lowpass = LowPassWindowedSinc()

    def forward(self, x):
        shape = list(x.shape)
        # print(shape)
        new_shape = shape[:-2] + [shape[-2] * self.ratio
                                  ] + [shape[-1] * self.ratio]

        x = x[..., 1::self.ratio, 1::self.ratio] #adding the 1 removes the shift
        x = self.lowpass(x)

        # print("xx shape", xx.shape)
        return x
