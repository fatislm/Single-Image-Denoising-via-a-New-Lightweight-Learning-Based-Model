import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class GaussianSmoothLayer(nn.Module):
    def __init__(self, channel, kernel_size, sigma, dim=2):
        super(GaussianSmoothLayer, self).__init__()
        kernel_x = cv2.getGaussianKernel(kernel_size, sigma)
        kernel_y = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel_x * kernel_y.T
        self.kernel_data = kernel
        self.groups = channel
        if dim == 1:
            self.conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                  groups=channel, bias=False)
        elif dim == 2:
            self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                  groups=channel, bias=False)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                  groups=channel, bias=False)
        else:
            raise RuntimeError('input dim is not supported! Please check it!')

        self.conv.weight.requires_grad = False
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel))
        self.pad = int((kernel_size - 1) / 2)

    def forward(self, input):
        intdata = input
        intdata = F.pad(intdata, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        output = self.conv(intdata)
        return output
