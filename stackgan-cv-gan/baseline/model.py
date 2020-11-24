import torch
import torch.nn as nn
import torch.nn.parallel
from config import cfg


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# upscale the spatical size by a factor of 2


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        0
    )
