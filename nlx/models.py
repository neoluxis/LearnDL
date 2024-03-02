import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import numpy as np


def bilinear_kernel(in_channels, out_channels, kernel_size):
    """ return a bilinear filter tensor

    Args:
        in_channels: int, number of input channels
        out_channels: int, number of output channels
        kernel_size: int, size of the kernel

    Returns:
        weight: tensor, bilinear filter tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight


class FCN(nn.Module):
    def __init__(self, num_classes):
        pretrained_net = torchvision.models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(pretrained_net.children())[:-2])
        self.resnet18.add_module('final_conv', nn.Conv2d(
            512, num_classes, kernel_size=1))
        self.resnet18.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                            kernel_size=64, padding=16, stride=32))
        conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False)
        conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))

        W = bilinear_kernel(num_classes, num_classes, 64)
        self.resnet18.transpose_conv.weight.data.copy_(W)
        
    def forward(self, x):
        return self.resnet18(x)
    
