# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Common modules."""

import ast
import contextlib
import json
import math
import platform
import warnings
import zipfile
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import tensorflow as tf

# Import 'ultralytics' package or install if missing
try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    import os

    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from utils import TryExcept
from utils.dataloaders import exif_transpose, letterbox
from utils.general import (
    LOGGER,
    ROOT,
    Profile,
    check_requirements,
    check_suffix,
    check_version,
    colorstr,
    increment_path,
    is_jupyter,
    make_divisible,
    non_max_suppression,
    scale_boxes,
    xywh2xyxy,
    xyxy2xywh,
    yaml_load,
)
from utils.torch_utils import copy_attr, smart_inference_mode


def autopad(k, p=None, d=1):
    """
    Pads kernel to 'same' output shape, adjusting for optional dilation; returns padding size.

    `k`: kernel, `p`: padding, `d`: dilation.
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class DilatedCNN(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super(DilatedCNN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False),
                            nn.ReLU(),
                            # nn.MaxPool2d(2,2)
                            )           #16,16
        self.classifier = nn.Sequential(
                                        nn.ReLU(),
                                        # nn.Dropout(0.5),
                                        nn.Linear(512,10),
                                        nn.Softmax())
    
    def forward(self, x):
        print(x.shape)
        x = self.layer1(x)
        x = self.classifier(x)
        # print(x)
        return x
    
    # def ConvModule(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
    #     return nn.Sequential(nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False),
    #                         nn.ReLU(),
    #                         nn.MaxPool2d(2,2)
    #                         )

class DilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(DilatedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    # def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initializes a standard convolution layer with optional batch normalization and activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # print("self = ", self)
        # print("x.size() = ", x.size())
        # # x = x[:, :, :16, :16]
        # print("Kích thước sau khi cắt:", x.size())
        # target_size = 16
        # if (x.size(2) < 16 or x.size(3) < 16) & (x.size(2) > 16 or x.size(3) > 16):
        # # Lấy chiều cao và chiều rộng tối đa giữa target_size và kích thước hiện tại
        #     # Cắt tensor về kích thước target_size
        #     x = x[:, :, :target_size, :target_size]
        
        # print("Kích thước sau khi cắt:", x.size())
        """Applies a convolution followed by batch normalization and an activation function to the input tensor `x`."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        # print("x = ", x)
        """Applies a fused convolution and activation function to the input tensor `x`."""
        return self.act(self.conv(x))
    


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initializes a depth-wise convolution layer with optional activation; args: input channels (c1), output
        channels (c2), kernel size (k), stride (s), dilation (d), and activation flag (act).
        """

        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initializes a depth-wise transpose convolutional layer for YOLOv5; args: input channels (c1), output channels
        (c2), kernel size (k), stride (s), input padding (p1), output padding (p2).
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))

# class CBAM(nn.Module):
#     def __init__(self, c1,c2):
#         super(CBAM, self).__init__()
#         self.channel_attention = ChannelAttentionModule(c1)
#         self.spatial_attention = SpatialAttentionModule()

#     def forward(self, x):
#         out = self.channel_attention(x) * x
#         out = self.spatial_attention(out) * out
#         return out

# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3) 
#         #self.act=SiLU()
#         self.sigmoid = nn.Sigmoid()
#     def forward(self, x):
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out))
#         return out
    
# class ChannelAttentionModule(nn.Module):
#     def __init__(self, c1, reduction=16):
#         super(ChannelAttentionModule, self).__init__()
#         mid_channel = c1 // reduction
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.shared_MLP = nn.Sequential(
#             nn.Linear(in_features=c1, out_features=mid_channel),
#             nn.ReLU(),
#             nn.Linear(in_features=mid_channel, out_features=c1)
#         )
#         self.sigmoid = nn.Sigmoid()
#         #self.act=SiLU()
#     def forward(self, x):
#         avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
#         maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
#         return self.sigmoid(avgout + maxout)

class SENet(nn.Module):
    # https://github.com/ElijahKz/YoloV5-attention-Modules
    def __init__(self, c1,  ratio=16):

        super(SENet, self).__init__()
        #c*1*1
        # print("Đây là c1 = ", c1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // ratio, c1, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        # print("c1 = ", c1)
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        # print("Long = ",x * y.expand_as(x))
        return x * y.expand_as(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class CBAM(nn.Module):
    # ch_in, ch_out, shortcut, groups, expansion, ratio, kernel_size
    def __init__(self, c1, c2, kernel_size=3, shortcut=True, g=1, e=0.5, ratio=16):
        """
        Initialize the CBAM (Convolutional Block Attention Module) .

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            shortcut (bool): Whether to use a shortcut connection.
            g (int): Number of groups for grouped convolutions.
            e (float): Expansion factor for hidden channels.
            ratio (int): Reduction ratio for the hidden channels in the channel attention block.
        """
        super(CBAM, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.channel_attention = ChannelAttention(c2, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        # print("c2 = ", c2)
    def forward(self, x):
        """
        Forward pass of the CBAM .

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor after applying the CBAM bottleneck.
        """        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') 
            x2 = self.cv2(self.cv1(x))
            out = self.channel_attention(x2) * x2
            out = self.spatial_attention(out) * out
            return x + out if self.add else out

# ________________________________________________________________

class CBAMDW(nn.Module):
    # ch_in, ch_out, shortcut, groups, expansion, ratio, kernel_size
    def __init__(self, c1, c2, kernel_size=3, shortcut=True, g=1, e=0.5, ratio=16):
        """
        Initialize the CBAM (Convolutional Block Attention Module) .

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            shortcut (bool): Whether to use a shortcut connection.
            g (int): Number of groups for grouped convolutions.
            e (float): Expansion factor for hidden channels.
            ratio (int): Reduction ratio for the hidden channels in the channel attention block.
        """
        super(CBAMDW, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConv(c1, c_, 1, 1)
        self.cv2 = DWConv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2
        self.channel_attention = ChannelAttention(c2, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        # print("c2 = ", c2)
    def forward(self, x):
        """
        Forward pass of the CBAM .

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor after applying the CBAM bottleneck.
        """        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') 
            x2 = self.cv2(self.cv1(x))
            out = self.channel_attention(x2) * x2
            out = self.spatial_attention(out) * out
            return x + out if self.add else out

class Involution(nn.Module):

    def __init__(self, c1, c2, kernel_size, stride):
        """
        Initialize the Involution module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            kernel_size (int): Size of the involution kernel.
            stride (int): Stride for the involution operation.
        """
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.c1 = c1
        reduction_ratio = 1
        self.group_channels = 16
        self.groups = self.c1 // self.group_channels
        self.conv1 = Conv(
            c1, c1 // reduction_ratio, 1)
        self.conv2 = Conv(
            c1 // reduction_ratio,
            kernel_size ** 2 * self.groups,
            1, 1)

        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        """
        Forward pass of the Involution module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            out (torch.Tensor): Output tensor after applying the involution operation.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') 
            weight = self.conv2(x)
            b, c, h, w = weight.shape
            weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
            out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
            out = (weight * out).sum(dim=3).view(b, self.c1, h, w)

            return out
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1) -> None:
        super(BasicBlock, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out
    
class ASPP(nn.Module):
    # Atrous Spatial Pyramid Pooling (ASPP) layer 
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)    
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.m = nn.ModuleList([nn.Conv2d(c_, c_, kernel_size=3, stride=1, padding=(x-1)//2, dilation=(x-1)//2, bias=False) for x in k])
        self.cv2 = Conv(c_ * (len(k) + 2), c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x]+ [self.maxpool(x)] + [m(x) for m in self.m] , 1)) 

class C3TR(nn.Module):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)
class C3STR(nn.Module):
    # C3 module with SwinTransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SwinTransformerBlock(c_, c_, c_//32, n)

class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.tr = nn.Sequential(*(SwinTransformerLayer(c2, num_heads=num_heads, window_size=window_size,  shift_size=0 if (i % 2 == 0) else self.shift_size ) for i in range(num_layers)))

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.tr(x)
        return x
    
# class InceptionDWConv2d(nn.Module):
#     """ Inception depthweise convolution
#     """
#     def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
#         super().__init__()
#         print("in_channels = ", in_channels)
#         gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
#         self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
#         self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
#         self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
#         self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
#     def forward(self, x):
#         print("x1 = ", x.shape)
#         x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
#         print("x_id = ", x_id)
#         print("self.dwconv_hw(x_hw) = ", self.dwconv_hw(x_hw))
#         print("x = ", x)
#         print("self.dwconv_h(x_h) = ", self.dwconv_h(x_h))
#         return torch.cat(
#             (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
#             dim=1
#         )

# class InceptionDWConv2d(nn.Module):
#     def __init__(self, c1, c2, k=3):
#         # print("c1InceptionDWConv2d = ", c1)
#         # print("c2InceptionDWConv2d = ", c2)
#         super().__init__()
#         c_ = c2 // 2
#         self.branch1 = nn.Sequential(
#             Conv(c1, c_, 1),
#             DWConv(c_, c_, k)
#         )
#         self.branch2 = nn.Sequential(
#             Conv(c1, c_, 1),
#             DWConv(c_, c_, k)
#         )
#         self.branch3 = nn.Sequential(
#             Conv(c1, c_, 1),
#             DWConv(c_, c_, k)
#         )
#         self.branch4 = nn.Sequential(
#             Conv(c1, c_, 1),
#             DWConv(c_, c_, k)
#         )

#     def forward(self, x):
#         # print(f"Input shape: {x.shape}")
#         # x = self.some_layer(x)
#         # print(f"After some_layer: {x.shape}")
#         # return torch.cat([self.branch1(x), self.branch2(x),self.branch3(x), self.branch4(x)], dim=1)
#         return torch.cat([self.branch1(x), self.branch2(x)], dim=1)

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        """
        Initializes a transformer layer, sans LayerNorm for performance, with multihead attention and linear layers.

        See  as described in https://arxiv.org/abs/2010.11929.
        """
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Performs forward pass using MultiheadAttention and two linear transformations with residual connections."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        """Initializes a Transformer block for vision tasks, adapting dimensions if necessary and stacking specified
        layers.
        """
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Processes input through an optional convolution, followed by Transformer layers and position embeddings for
        object detection.
        """
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

class C3_bottleneck_inception(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        self.in_channels = c1
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)        
        self.dwconv = wf_inception_module(c1)
        self.add = shortcut and c1 == c2

    def forward(self, x):

        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        print("x.shape = ", x.shape)
        # return x + self.cv2(self.m(self.cv1(x))) if self.add else self.cv2(self.cv1(x))
        y = self.cv1(x) + self.cv2(x)
        tuple = self.dwconv(y)

        return torch.cat([tuple], 1)
        # return x + torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Bottleneck2(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializefbs a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    


class Bottleneck2test(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # print("c1_ = ", c_)
        self.cv1 = Conv(c1, c_, 1, 1)
        # print("c2_ = ", c_)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.cbam = CBAM(c2, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        # return x
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        y = self.cv2(self.cv1(x))
        y = self.cbam(y)

        return x + y if self.add else y
        # return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

class Bottleneck1CBAM(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # print("c1_ = ", c_)
        self.cv1 = Conv(c1, c_, 1, 1)
        # print("c2_ = ", c_)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # self.wf_inception_module = wf_inception_module(c2, c2)
        # self.stem = STEM(c2, c2)
        self.cbam = CBAM(c2, c2)
        self.SE_Block = SE_Block(c2, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        # return x
        # return x + self.cv2(self.cv1(x)) if Bottleneckself.add else self.cv2(self.cv1(x))
        y = self.cv2(self.cv1(x))
        # y = self.wf_inception_module(y)
        y = self.cbam(y)
        y = self.SE_Block(y)
        
        return x + y if self.add else y

class Bottleneck1CBAM_wf_inception_module(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # print("c1_ = ", c_)
        self.cv1 = Conv(c1, c_, 1, 1)
        # print("c2_ = ", c_)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # self.wf_inception_module = wf_inception_module(c2, c2)
        self.stem = STEM(c2, c2)
        self.wf_inception_module = wf_inception_module(c1)
        self.cbam = CBAM(c2, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        # return x
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        y = self.cv2(self.cv1(x))
        y = self.stem(y)
        y = self.wf_inception_module(y)
        y = self.cbam(y)
        
        return x + y if self.add else y

class C3ICSs(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 // 2)  # hidden channels

        self.conv = nn.Sequential(
            Conv(c1, c_, 1, 1),
            Conv(c_, c2, 3, 1, g=g),
            wf_inception_module(c1),
            CBAM(c2, c2),
            # SE_Block(c2, c2),
            Bottleneck(c_, c_, shortcut, g, e=1.0)
        )

        # self.shortcut = nn.Sequential(*( for _ in range(n)))

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        # return x
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        # y = self.cv2(self.cv1(x))
        # y = self.wf_inception_module(y)
        # y = self.CBAM(y)
        # y = self.SE_Block(y)
        # # y = self.m(y)
        # print("y.shape() = ", y.shape)
        # return self.cv3(torch.cat(self.cv1(y), self.cv2(x)), 1)
        return self.conv(x)

class C3ICS(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        # self.stem = STEM(c2, c2)
        self.inception = wf_inception_module(c1)
        # self.SeparableConv2d = SeparableConv2d(c2, c2,3)
        self.CBAM = CBAM(c2, c2)
        # self.SE_Block = SE_Block(c2, c2),
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        # y = self.stem(x)
        y = self.inception(x)
        # y = self.SeparableConv2d(y)
        y = self.CBAM(y)
        # y = self.SE_Block(y)
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)) + y


class Bottleneck1CBAM_wfinception_CBAM(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # print("c1_ = ", c_)
        self.cv1 = Conv(c1, c_, 1, 1)
        # print("c2_ = ", c_)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        # self.stem = STEM(c2, c2)
        self.wf_inception_module = wf_inception_module(c2)
        # self.CBAM = CBAM(c2, c2)
        # self.SE_Block = SE_Block(c_, c_)
        self.m = Bottleneck(c_, c_, shortcut, g, e=1.0)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        # return x
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        # y = self.cv2(self.cv1(x))
        # y = self.wf_inception_module(y)
        # y = self.CBAM(y)
        # y = self.SE_Block(y)

        y = self.m(self.wf_inception_module(self.cv2(self.cv1(x))))

        return x + y if self.add else y
    
class Bottleneck1CBAMSE_Block(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # print("c1_ = ", c_)
        self.cv1 = Conv(c1, c_, 1, 1)
        # print("c2_ = ", c_)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # self.wf_inception_module = wf_inception_module(c2, c2)
        # self.stem = STEM(c2, c2)
        self.cbam = CBAM(c2, c2)
        self.SE_Block = SE_Block(c2, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        # return x
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        y = self.cv2(self.cv1(x))
        # y = self.wf_inception_module(y)
        y = self.cbam(y)
        y = self.SE_Block(y)
        
        return x + y if self.add else y
    
class Bottleneck1CBAMDW(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # print("c1_ = ", c_)
        self.cv1 = DWConv(c1, c_, 1, 1)
        # print("c2_ = ", c_)
        self.cv2 = DWConv(c_, c2, 3, 1)
        # self.wf_inception_module = wf_inception_module(c2, c2)
        # self.stem = STEM(c2, c2)
        self.cbam = CBAM(c2, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        # return x
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        y = self.cv2(self.cv1(x))
        # y = self.wf_inception_module(y)
        y = self.cbam(y)
        
        return x + y if self.add else y

class CSP1CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck1CBAM(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class CSP2CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels        
        self.cv1 = Conv(c1, c_, 1, 1)
        # print("c1_ = ", c_)
        self.cv2 = Conv(c1, c_, 1, 1)
        # print("c2_ = ", c_)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        # print("c3_ = ", c_)
        self.m = nn.Sequential(*(Bottleneck2(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        c_ = x.shape[1]
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        print("Shape of concat1f:", self.m(self.cv1(x)).shape)
        print("Shape of concat2f:", self.cv2(x).shape)
        result = self.cv3(torch.cat((self.m(self.cv1(x)[:, :c_]), self.cv2(x)[:, :c_]), 1))
        print("result = ", result.shape)
        return result


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP bottleneck with optional shortcuts; args: ch_in, ch_out, number of repeats, shortcut bool,
        groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward pass by applying layers, activation, and concatenation on input x, returning feature-
        enhanced output.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class BottleneckCSPAdvanced(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP bottleneck with optional shortcuts; args: ch_in, ch_out, number of repeats, shortcut bool,
        groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        # self.m = nn.Sequential(*(Bottleneck1CBAM(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        # self.m = nn.Sequential(*(Bottleneck1CBAMSE_Block(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck1CBAM_wf_inception_module(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward pass by applying layers, activation, and concatenation on input x, returning feature-
        enhanced output.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class BottleneckCSPAdvancedDW(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes CSP bottleneck with optional shortcuts; args: ch_in, ch_out, number of repeats, shortcut bool,
        groups, expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DWConv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = DWConv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck1CBAMDW(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Performs forward pass by applying layers, activation, and concatenation on input x, returning feature-
        enhanced output.
        """
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        """
        Initializes CrossConv with downsampling, expanding, and optionally shortcutting; `c1` input, `c2` output
        channels.

        Inputs are ch_in, ch_out, kernel, stride, groups, expansion, shortcut.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Performs feature sampling, expanding, and applies shortcut if channels match; expects `x` input tensor."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # print("CABottleneck c_ = ", c_)
        # print("CABottleneck c1 = ", c1)
        # print("CABottleneck c2 = ", c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        # print("c3 = ", x.shape)
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class CABottleneck(nn.Module):
    #Custom CA Fadil
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=32):
        super().__init__()
        c_ = int(c2 * e)
        # print("CABottleneck c_ = ", c_)
        # print("CABottleneck c1 = ", c1)
        # print("CABottleneck c2 = ", c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.ca = CoordAtt(c1,c2,ratio)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, c1 // ratio)
        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1=self.cv2(self.cv1(x))
        n, c, h, w = x.size()
        x_h = self.pool_h(x1)
        x_w = self.pool_w(x1).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h,w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = x1 * a_w * a_h
        # print("c3ca = ", x.shape)
        return x + out if self.add else out

class CACBAMBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        """Initializes a standard bottleneck layer with optional shortcut and group convolution, supporting channel
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # print("c1_ = ", c_)
        self.cv1 = Conv(c1, c_, 1, 1)
        # print("c2_ = ", c_)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # self.wf_inception_module = wf_inception_module(c2, c2)
        self.CABottleneck = CABottleneck(c2, c2)
        self.cbam = CBAM(c2, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Processes input through two convolutions, optionally adds shortcut if channel dimensions match; input is a
        tensor.
        """
        # return x
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        y = self.cv2(self.cv1(x))
        y = self.CABottleneck(y)
        y = self.cbam(y)
        
        return x + y if self.add else y
class C3CA(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CABottleneck(c_, c_, shortcut) for _ in range(n))) 

class C3CACBAM(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CACBAMBottleneck(c_, c_, shortcut) for _ in range(n))) 

class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3x module with cross-convolutions, extending C3 with customizable channel dimensions, groups,
        and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with TransformerBlock for enhanced feature extraction, accepts channel sizes, shortcut
        config, group, and expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        """Initializes a C3 module with SPP layer for advanced spatial feature extraction, given channel sizes, kernel
        sizes, shortcut, group, and expansion ratio.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes YOLOv5's C3 module with Ghost Bottlenecks for efficient feature extraction."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initializes SPP layer with Spatial Pyramid Pooling, ref: https://arxiv.org/abs/1406.4729, args: c1 (input channels), c2 (output channels), k (kernel sizes)."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Applies convolution and max pooling layers to the input tensor `x`, concatenates results, and returns output
        tensor.
        """
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        # print('SPPFPc_ = ', c_)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        # print('SPPFPc2 = ', c2)
        # print('self.cv1 = ', self.cv1)
        # print('self.cv2 = ', self.cv2)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # print('self.m = ', self.m)

    def forward(self, x):
        # print("ở đây ")
        # print("self = ", self)
        """Processes input through a series of convolutions and max pooling operations for feature extraction."""
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class SPPFP(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):
        """
        Initializes YOLOv5 SPPF layer with given channels and kernel size for YOLOv5 model, combining convolution and
        max pooling.

        Equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        # print('SPPFPc_ = ', c_)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        # print('SPPFPc2 = ', c2)
        # print('self.cv1 = ', self.cv1)
        # print('self.cv2 = ', self.cv2)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        # print('self.m = ', self.m)

    def forward(self, x):
        """Processes input through a series of convolutions and max pooling operations with multiple concatenation steps for feature extraction."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # print("Shape of x:", x.shape)
        # print("Shape of y1:", y1.shape)
        # print("Shape of y2:", y2.shape)
        # print("Shape of y3:", y3.shape)
        
        # Concatenations according to the diagram
        # concat1a = torch.cat((x, y1), 1)
        # print("Shape of concat1a:", concat1a.shape)
        # concat2a = torch.cat((concat1a, y2, x), 1)
        # print("Shape of concat2a:", concat2a.shape)
        # concat3a = torch.cat((concat2a, y3, x), 1)
        # print("Shape of concat3a:", concat3a.shape)

        # final_concata = torch.cat((x, concat1a, concat2a, concat3a), 1)
        # print("Shape of finala concatenation:", final_concata.shape)

        # resulta = self.cv2(final_concata)
        # print("Shape of outputa:", resulta.shape)
        
        # return result
    
        # c_ = x.shape[1] // 2
        # concat1 = torch.cat((x[:, :c_], y1[:, :c_]), 1)
        # print("Shape of concat1:", concat1.shape)
        # concat2 = torch.cat((concat1[:, :c_], y2[:, :c_], x[:, :c_]), 1)
        # print("Shape of concat2:", concat2.shape)
        # concat3 = torch.cat((concat2[:, :c_], y3[:, :c_], x[:, :c_]), 1)
        # print("Shape of concat3:", concat3.shape)

        # final_concat = torch.cat((x, concat1, concat2, concat3), 1)
        # print("Shape of final concatenation:", final_concat.shape)

        # result = self.cv2(final_concat)
        # print("Shape of output:", result.shape)
        
        # return result
        c_ = x.shape[1]
        concat1 = torch.cat([x[:, :c_], y1[:, :c_]], 1)
        # print("Shape of concat1:", concat1.shape)
        concat2 = torch.cat([concat1[:, :c_], y2[:, :c_], x[:, :c_]], 1)
        # print("Shape of concat2:", concat2.shape)
        concat3 = torch.cat([concat1[:, :c_], concat2[:, :c_], y3[:, :c_], x[:, :c_]], 1)
        # print("Shape of concat3:", concat3.shape)
        result = torch.cat([x[:, :c_ ],concat1[:, :c_], concat2[:, :c_], concat3[:, :c_]], 1)
        # print("Shape of result:", result.shape)
        return self.cv2(result)

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus module to concentrate width-height info into channel space with configurable convolution
        parameters.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Processes input through Focus mechanism, reshaping (b,c,w,h) to (b,4c,w/2,h/2) then applies convolution."""
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes GhostConv with in/out channels, kernel size, stride, groups, and activation; halves out channels
        for efficiency.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Performs forward pass, concatenating outputs of two convolutions on input `x`: shape (B,C,H,W)."""
        y = self.cv1(x)
        # print("y = ", y.shape)
        return torch.cat((y, self.cv2(y)), 1)


# class GhostBottleneck(nn.Module):
#     # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
#     def __init__(self, c1, c2, k=3, s=1):
#         """Initializes GhostBottleneck with ch_in `c1`, ch_out `c2`, kernel size `k`, stride `s`; see https://github.com/huawei-noah/ghostnet."""
#         super().__init__()
#         c_ = c2 // 2
#         self.conv = nn.Sequential(
#             GhostConv(c1, c_, 1, 1),  # pw
#             DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
#             GhostConv(c_, c2, 1, 1, act=False),
#         )  # pw-linear
#         self.shortcut = (
#             nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
#         )

#     def forward(self, x):
#         """Processes input through conv and shortcut layers, returning their summed output."""
#         print("x.shape = ", x.shape)
#         return self.conv(x) + self.shortcut(x)
    
class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):
        """Initializes 
         
           with ch_in `c1`, ch_out `c2`, kernel size `k`, stride `s`; see https://github.com/huawei-noah/ghostnet."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Processes input through conv and shortcut layers, returning their summed output."""
        return self.conv(x) + self.shortcut(x)
    
class GhostBottleneckCBAM(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):
        """Initializes 
         
           with ch_in `c1`, ch_out `c2`, kernel size `k`, stride `s`; see https://github.com/huawei-noah/ghostnet."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
            wf_inception_module(c1),
            SeparableConv2d(c2,c2,3),
            CBAM(c2, c2)
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Processes input through conv and shortcut layers, returning their summed output."""
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        """Initializes a layer to contract spatial dimensions (width-height) into channels, e.g., input shape
        (1,64,80,80) to (1,256,40,40).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor to expand channel dimensions by contracting spatial dimensions, yielding output shape
        `(b, c*s*s, h//s, w//s)`.
        """
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        """
        Initializes the Expand module to increase spatial dimensions by redistributing channels, with an optional gain
        factor.

        Example: x(1,64,80,80) to x(1,16,160,160).
        """
        super().__init__()
        self.gain = gain

    def forward(self, x):
        """Processes input tensor x to expand spatial dimensions by redistributing channels, requiring C / gain^2 ==
        0.
        """
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)

def resize_if_smaller(x):
    """
    Chuyển kích thước của tensor `x` về `target_size` nếu chiều cao hoặc chiều rộng nhỏ hơn `target_size`.
    
    Args:
    - x (torch.Tensor): Tensor cần thay đổi kích thước.
    - target_size (int): Kích thước mục tiêu mà chiều cao và chiều rộng cần chuyển đổi đến.
    
    Returns:
    - torch.Tensor: Tensor đã được thay đổi kích thước.
    """
    target_size = 16
    if x.size(2) < target_size or x.size(3) < target_size:
        # Lấy chiều cao và chiều rộng tối đa giữa target_size và kích thước hiện tại
        new_size = max(target_size, x.size(2), x.size(3))
        
        # Sử dụng hàm interpolate để thay đổi kích thước
        x = torch.nn.functional.interpolate(x, size=(new_size, new_size), mode='nearest')
    
    return x[:, :, :target_size, :target_size]

# class Concat(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, dimension=1):
#         """Initializes a Concat module to concatenate tensors along a specified dimension."""
#         super().__init__()
#         self.d = dimension

#     def forward(self, x):
#         """Concatenates a list of tensors along a specified dimension; `x` is a list of tensors, `dimension` is an
#         int.
#         """
#         # c_ = x.shape[1]
#         print(f"Input shape Concat: {len(x)}")
#         # In ra kích thước của từng Tensor trong danh sách `x`
#         # for i, tensor in enumerate(x):
#         #     print(f"Input tensor {i} shape: {tensor.shape}")
#         #     x = torch.cat(x[:, :tensor.shape], self.d)

        
#         # x = torch.cat(x[:, :c_], self.d)
#         x = torch.cat(x, self.d)
#         print(f"Input shape Concat: {x.shape}")
#         return x

import torch
import torch.nn as nn

class Concat(nn.Module):
    # Concatenate a list of tensors along a specified dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        # Kiểm tra kích thước của mỗi tensor
        # for i, tensor in enumerate(x):
        #     print(f"Input tensor {i} shape: {tensor.shape}")
        
        # Điều chỉnh kích thước tensor nếu cần thiết (ví dụ sử dụng nn.Upsample hoặc nn.Conv2d)
        # Giả sử chúng ta muốn làm cho tất cả tensor có cùng kích thước với tensor đầu tiên
        target_size = x[0].shape[-2:]  # Lấy kích thước không gian của tensor đầu tiên (H, W)

        resized_tensors = []
        for tensor in x:
            if tensor.shape[-2:] != target_size:
                tensor = nn.functional.interpolate(tensor, size=target_size, mode='nearest')  # Upsample hoặc Downsample
            resized_tensors.append(tensor)
        
        # Sau khi điều chỉnh kích thước, thực hiện nối
        x = torch.cat(resized_tensors, self.d)
        # print(f"Output shape after Concat: {x.shape}")
        return x

    
# def forward(self, x):
#     c_ = x.shape[1]
#     """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
#     print("Shape of concat1f:", self.m(self.cv1(x)).shape)
#     print("Shape of concat2f:", self.cv2(x).shape)
#     result = self.cv3(torch.cat((self.m(self.cv1(x)[:, :c_]), self.cv2(x)[:, :c_]), 1))
#     print("result = ", result.shape)
#     return result

    # Concatenate a list of tensors along dimension
    # def __init__(self, c1, c2):
    #     # print("c1 = ", c1)
    #     # print("c2 = ", c2)
    #     super(Concat, self).__init__()
    #     # self.relu = nn.ReLU()
    #     self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    #     self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
    #     self.epsilon = 0.0001
    #     self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
    #     self.swish = MemoryEfficientSwish()
    #     self.forward_count = 0  # Thêm biến đếm này
    #     self.adjust_c1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
    # def forward(self, x):
    #     outs = self._forward(x)
    #     return outs
    
    # def _forward(self, x):

    #     self.forward_count += 1  # Tăng biến đếm mỗi khi _forward được gọi
    #     if len(x) == 2:
    #         # w = self.relu(self.w1)
    #         w = self.w1
    #         weight = w / (torch.sum(w, dim=0) + self.epsilon)
    #         # Connections for P6_0 and P7_0 to P6_1 respectively
    #         # print("weight[0].shape, x[0].shape = ", weight[0].shape, x[0].shape)
    #         # print("weight[1].shape, x[1].shape = ", weight[1].shape, x[1].shape)
    #         x1 = x[1].shape
    #         channel_size = x1[1]  # Lấy phần tử thứ hai (index 1)
    #         # print(channel_size)  # Sẽ in ra 256
    #         x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1]))
    #         # return x
    #     elif len(x) == 3:
    #         # w = self.relu(self.w2)
    #         w = self.w2
    #         weight = w / (torch.sum(w, dim=0) + self.epsilon)
    #         x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
        
    #     # print(f"_forward has been called {self.forward_count} times")
    #     # print("x.shape = ", x.shape)
    #     return x

    # def _forward(self, x):
    #     if len(x) == 2:
    #         w = self.w1
    #         weight = w / (torch.sum(w, dim=0) + self.epsilon)
            
    #         # Điều chỉnh số kênh của x[0] nếu cần
    #         if x[0].shape[1] != x[1].shape[1]:
    #             x0_adjusted = self.adjust_c1(x[0])
    #         else:
    #             x0_adjusted = x[0]
            
    #         x = self.conv(self.swish(weight[0] * x0_adjusted + weight[1] * x[1]))
    #     elif len(x) == 3:
    #         w = self.w2
    #         weight = w / (torch.sum(w, dim=0) + self.epsilon)
            
    #         # Điều chỉnh số kênh của x[0] và x[1] nếu cần
    #         if x[0].shape[1] != x[2].shape[1]:
    #             x0_adjusted = self.adjust_c1(x[0])
    #         else:
    #             x0_adjusted = x[0]
            
    #         if x[1].shape[1] != x[2].shape[1]:
    #             x1_adjusted = self.adjust_c1(x[1])
    #         else:
    #             x1_adjusted = x[1]
            
    #         x = self.conv(self.swish(weight[0] * x0_adjusted + weight[1] * x1_adjusted + weight[2] * x[2]))

    #     return x


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights="yolov5s.pt", device=torch.device("cpu"), dnn=False, data=None, fp16=False, fuse=True):
        """Initializes DetectMultiBackend with support for various inference backends, including PyTorch and ONNX."""
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import attempt_download, attempt_load  # scoped to avoid circular import

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        if not (pt or triton):
            w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            stride = max(int(model.stride.max()), 32)  # model stride
            names = model.module.names if hasattr(model, "module") else model.names  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            LOGGER.info(f"Loading {w} for TorchScript inference...")
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                d = json.loads(
                    extra_files["config.txt"],
                    object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                )
                stride, names = int(d["stride"]), d["names"]
        elif dnn:  # ONNX OpenCV DNN
            LOGGER.info(f"Loading {w} for ONNX OpenCV DNN inference...")
            check_requirements("opencv-python>=4.5.4")
            net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            LOGGER.info(f"Loading {w} for ONNX Runtime inference...")
            check_requirements(("onnx", "onnxruntime-gpu" if cuda else "onnxruntime"))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if "stride" in meta:
                stride, names = int(meta["stride"]), eval(meta["names"])
        elif xml:  # OpenVINO
            LOGGER.info(f"Loading {w} for OpenVINO inference...")
            check_requirements("openvino>=2023.0")  # requires openvino-dev: https://pypi.org/project/openvino-dev/
            from openvino.runtime import Core, Layout, get_batch

            core = Core()
            if not Path(w).is_file():  # if not *.xml
                w = next(Path(w).glob("*.xml"))  # get *.xml file from *_openvino_model dir
            ov_model = core.read_model(model=w, weights=Path(w).with_suffix(".bin"))
            if ov_model.get_parameters()[0].get_layout().empty:
                ov_model.get_parameters()[0].set_layout(Layout("NCHW"))
            batch_dim = get_batch(ov_model)
            if batch_dim.is_static:
                batch_size = batch_dim.get_length()
            ov_compiled_model = core.compile_model(ov_model, device_name="AUTO")  # AUTO selects best available device
            stride, names = self._load_metadata(Path(w).with_suffix(".yaml"))  # load metadata
        elif engine:  # TensorRT
            LOGGER.info(f"Loading {w} for TensorRT inference...")
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):  # dynamic
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_profile_shape(name, 0)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_tensor_shape(name))
                else:
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                        if dtype == np.float16:
                            fp16 = True
                    else:  # output
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        elif coreml:  # CoreML
            LOGGER.info(f"Loading {w} for CoreML inference...")
            import coremltools as ct

            model = ct.models.MLModel(w)
        elif saved_model:  # TF SavedModel
            LOGGER.info(f"Loading {w} for TensorFlow SavedModel inference...")
            import tensorflow as tf

            keras = False  # assume TF1 saved_model
            model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
        elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            LOGGER.info(f"Loading {w} for TensorFlow GraphDef inference...")
            import tensorflow as tf

            def wrap_frozen_graph(gd, inputs, outputs):
                """Wraps a TensorFlow GraphDef for inference, returning a pruned function."""
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
                ge = x.graph.as_graph_element
                return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))

            def gd_outputs(gd):
                """Generates a sorted list of graph outputs excluding NoOp nodes and inputs, formatted as '<name>:0'."""
                name_list, input_list = [], []
                for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
                    name_list.append(node.name)
                    input_list.extend(node.input)
                return sorted(f"{x}:0" for x in list(set(name_list) - set(input_list)) if not x.startswith("NoOp"))

            gd = tf.Graph().as_graph_def()  # TF GraphDef
            with open(w, "rb") as f:
                gd.ParseFromString(f.read())
            frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs=gd_outputs(gd))
        elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import Interpreter, load_delegate
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                LOGGER.info(f"Loading {w} for TensorFlow Lite Edge TPU inference...")
                delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                    platform.system()
                ]
                interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
            else:  # TFLite
                LOGGER.info(f"Loading {w} for TensorFlow Lite inference...")
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta["stride"]), meta["names"]
        elif tfjs:  # TF.js
            raise NotImplementedError("ERROR: YOLOv5 TF.js inference is not supported")
        elif paddle:  # PaddlePaddle
            LOGGER.info(f"Loading {w} for PaddlePaddle inference...")
            check_requirements("paddlepaddle-gpu" if cuda else "paddlepaddle")
            import paddle.inference as pdi

            if not Path(w).is_file():  # if not *.pdmodel
                w = next(Path(w).rglob("*.pdmodel"))  # get *.pdmodel file from *_paddle_model dir
            weights = Path(w).with_suffix(".pdiparams")
            config = pdi.Config(str(w), str(weights))
            if cuda:
                config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            predictor = pdi.create_predictor(config)
            input_handle = predictor.get_input_handle(predictor.get_input_names()[0])
            output_names = predictor.get_output_names()
        elif triton:  # NVIDIA Triton Inference Server
            LOGGER.info(f"Using {w} as Triton Inference Server...")
            check_requirements("tritonclient[all]")
            from utils.triton import TritonRemoteModel

            model = TritonRemoteModel(url=w)
            nhwc = model.runtime.startswith("tensorflow")
        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")

        # class names
        if "names" not in locals():
            names = yaml_load(data)["names"] if data else {i: f"class{i}" for i in range(999)}
        if names[0] == "n01440764" and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / "data/ImageNet.yaml")["names"]  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        """Performs YOLOv5 inference on input images with options for augmentation and visualization."""
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.ov_compiled_model(im).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.BILINEAR)
            y = self.model.predict({"image": im})  # coordinates are xywh normalized
            if "confidence" in y:
                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y["confidence"].max(1), y["confidence"].argmax(1).astype(np.float)
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(reversed(y.values()))  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input["quantization"]
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if int8:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        """Converts a NumPy array to a torch tensor, maintaining device compatibility."""
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640)):
        """Performs a single inference warmup to initialize model weights, accepting an `imgsz` tuple for image size."""
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        """
        Determines model type from file path or URL, supporting various export formats.

        Example: path='path/to/model.onnx' -> type=onnx
        """
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        from export import export_formats
        from utils.downloads import is_url

        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        """Loads metadata from a YAML file, returning strides and names if the file exists, otherwise `None`."""
        if f.exists():
            d = yaml_load(f)
            return d["stride"], d["names"]  # assign stride, names
        return None, None


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.55  # NMS confidence threshold
    iou = 0.65  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model, verbose=True):
        """Initializes YOLOv5 model for inference, setting up attributes and preparing model for evaluation."""
        super().__init__()
        if verbose:
            LOGGER.info("Adding AutoShape... ")
        copy_attr(self, model, include=("yaml", "nc", "hyp", "names", "stride", "abc"), exclude=())  # copy attributes
        self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.inplace = False  # Detect.inplace=False for safe multithread inference
            m.export = True  # do not output loss values

    def _apply(self, fn):
        """
        Applies to(), cpu(), cuda(), half() etc.

        to model tensors excluding parameters or registered buffers.
        """
        self = super()._apply(fn)
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @smart_inference_mode()
    def forward(self, ims, size=640, augment=False, profile=False):
        """
        Performs inference on inputs with optional augment & profiling.

        Supports various formats including file, URI, OpenCV, PIL, numpy, torch.
        """
        # For size(height=640, width=1280), RGB images example inputs are:
        #   file:        ims = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        dt = (Profile(), Profile(), Profile())
        with dt[0]:
            if isinstance(size, int):  # expand
                size = (size, size)
            p = next(self.model.parameters()) if self.pt else torch.empty(1, device=self.model.device)  # param
            autocast = self.amp and (p.device.type != "cpu")  # Automatic Mixed Precision (AMP) inference
            if isinstance(ims, torch.Tensor):  # torch
                with amp.autocast(autocast):
                    return self.model(ims.to(p.device).type_as(p), augment=augment)  # inference

            # Pre-process
            n, ims = (len(ims), list(ims)) if isinstance(ims, (list, tuple)) else (1, [ims])  # number, list of images
            shape0, shape1, files = [], [], []  # image and inference shapes, filenames
            for i, im in enumerate(ims):
                f = f"image{i}"  # filename
                if isinstance(im, (str, Path)):  # filename or uri
                    im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith("http") else im), im
                    im = np.asarray(exif_transpose(im))
                elif isinstance(im, Image.Image):  # PIL Image
                    im, f = np.asarray(exif_transpose(im)), getattr(im, "filename", f) or f
                files.append(Path(f).with_suffix(".jpg").name)
                if im.shape[0] < 5:  # image in CHW
                    im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
                im = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # enforce 3ch input
                s = im.shape[:2]  # HWC
                shape0.append(s)  # image shape
                g = max(size) / max(s)  # gain
                shape1.append([int(y * g) for y in s])
                ims[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
            shape1 = [make_divisible(x, self.stride) for x in np.array(shape1).max(0)]  # inf shape
            x = [letterbox(im, shape1, auto=False)[0] for im in ims]  # pad
            x = np.ascontiguousarray(np.array(x).transpose((0, 3, 1, 2)))  # stack and BHWC to BCHW
            x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32

        with amp.autocast(autocast):
            # Inference
            with dt[1]:
                y = self.model(x, augment=augment)  # forward

            # Post-process
            with dt[2]:
                y = non_max_suppression(
                    y if self.dmb else y[0],
                    self.conf,
                    self.iou,
                    self.classes,
                    self.agnostic,
                    self.multi_label,
                    max_det=self.max_det,
                )  # NMS
                for i in range(n):
                    scale_boxes(shape1, y[i][:, :4], shape0[i])

            return Detections(ims, y, files, dt, self.names, x.shape)


class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, ims, pred, files, times=(0, 0, 0), names=None, shape=None):
        """Initializes the YOLOv5 Detections class with image info, predictions, filenames, timing and normalization."""
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in ims]  # normalizations
        self.ims = ims  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple(x.t / self.n * 1e3 for x in times)  # timestamps (ms)
        self.s = tuple(shape)  # inference BCHW shape

    def _run(self, pprint=False, show=False, save=False, crop=False, render=False, labels=True, save_dir=Path("")):
        """Executes model predictions, displaying and/or saving outputs with optional crops and labels."""
        s, crops = "", []
        for i, (im, pred) in enumerate(zip(self.ims, self.pred)):
            s += f"\nimage {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} "  # string
            if pred.shape[0]:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                s = s.rstrip(", ")
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f"{self.names[int(cls)]} {conf:.2f}"
                        if crop:
                            file = save_dir / "crops" / self.names[int(cls)] / self.files[i] if save else None
                            crops.append(
                                {
                                    "box": box,
                                    "conf": conf,
                                    "cls": cls,
                                    "label": label,
                                    "im": save_one_box(box, im, file=file, save=save),
                                }
                            )
                        else:  # all others
                            annotator.box_label(box, label if labels else "", color=colors(cls))
                    im = annotator.im
            else:
                s += "(no detections)"

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if show:
                if is_jupyter():
                    from IPython.display import display

                    display(im)
                else:
                    im.show(self.files[i])
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.ims[i] = np.asarray(im)
        if pprint:
            s = s.lstrip("\n")
            return f"{s}\nSpeed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {self.s}" % self.t
        if crop:
            if save:
                LOGGER.info(f"Saved results to {save_dir}\n")
            return crops

    @TryExcept("Showing images is not supported in this environment")
    def show(self, labels=True):
        """
        Displays detection results with optional labels.

        Usage: show(labels=True)
        """
        self._run(show=True, labels=labels)  # show results

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Saves detection results with optional labels to a specified directory.

        Usage: save(labels=True, save_dir='runs/detect/exp', exist_ok=False)
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        """
        Crops detection results, optionally saves them to a directory.

        Args: save (bool), save_dir (str), exist_ok (bool).
        """
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        """Renders detection results with optional labels on images; args: labels (bool) indicating label inclusion."""
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def pandas(self):
        """
        Returns detections as pandas DataFrames for various box formats (xyxy, xyxyn, xywh, xywhn).

        Example: print(results.pandas().xyxy[0]).
        """
        new = copy(self)  # return copy
        ca = "xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"  # xyxy columns
        cb = "xcenter", "ycenter", "width", "height", "confidence", "class", "name"  # xywh columns
        for k, c in zip(["xyxy", "xyxyn", "xywh", "xywhn"], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        """
        Converts a Detections object into a list of individual detection results for iteration.

        Example: for result in results.tolist():
        """
        r = range(self.n)  # iterable
        return [
            Detections(
                [self.ims[i]],
                [self.pred[i]],
                [self.files[i]],
                self.times,
                self.names,
                self.s,
            )
            for i in r
        ]

    def print(self):
        """Logs the string representation of the current object's state via the LOGGER."""
        LOGGER.info(self.__str__())

    def __len__(self):
        """Returns the number of results stored, overrides the default len(results)."""
        return self.n

    def __str__(self):
        """Returns a string representation of the model's results, suitable for printing, overrides default
        print(results).
        """
        return self._run(pprint=True)  # print results

    def __repr__(self):
        """Returns a string representation of the YOLOv5 object, including its class and formatted results."""
        return f"YOLOv5 {self.__class__} instance\n" + self.__str__()


class Proto(nn.Module):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):
        """Initializes YOLOv5 Proto module for segmentation with input, proto, and mask channels configuration."""
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass using convolutional layers and upsampling on input tensor `x`."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0
    ):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Processes input through conv, pool, drop, and linear layers; supports list concatenation input."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# ====================================================================

# config = [
#     [-1, 1, Conv, [64, 6, 2, 2]],             # 0-P1/2
#     [-1, 1, Conv, [128, 3, 2]],               # 1-P2/4
#     [-1, 3, C3, [128]],                      # 2

#     [-1, 1, Conv, [256, 3, 2]],               # 3-P3/8
#     [-1, 6, C3, [256]],                      # 4
#     [-1, 1, DWConv, [256, 3]],               # 5
#     [-1, 1, InceptionDWConv2d, [128, 128, 3]],# 6

#     [-1, 1, Conv, [512, 3, 2]],               # 7-P4/16
#     [-1, 9, C3, [512]],                      # 8
#     [-1, 1, InceptionDWConv2d, [256, 256, 3]],# 9

#     [-1, 1, Conv, [1024, 3, 2]],              # 10-P5/32
#     [-1, 3, C3, [1024]],                     # 11
#     [-1, 1, InceptionDWConv2d, [512, 512, 3]],# 12

#     [-1, 1, SPPF, [1024, 5]],                # 13
# ]

# # Hàm xây dựng mô hình dựa trên cấu hình
# def build_model(config):
#     layers = []
#     current_tensor = input_tensor

#     for i, module in enumerate(config):
#         from_idx, num, module_type, args = module
        
#         # Tạo lớp mạng từ module_type và args
#         module = module_type(*args)
        
#         # Thực hiện truyền thuận của lớp mạng với số lần lặp num
#         for _ in range(num):
#             current_tensor = module(current_tensor)
#             layers.append(module)

#     return torch.nn.Sequential(*layers)

# # Khởi tạo mô hình
# model = build_model(config)

# # Kiểm tra kích thước của tensor sau mỗi mô-đun
# with torch.no_grad():
#     output = model(input_tensor)
#     print("Kích thước của tensor đầu ra:", output.size())

class wf_inception_module(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv_1 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size = 1, padding = 0, dilation=1, groups=in_channels//4)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.conv_1_3 = nn.Sequential(
                nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (1,3), padding = (0,1), dilation=1, groups=in_channels//4),
                nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (3,1), padding = (1,0), dilation=1, groups=in_channels//4)
        )      
        self.conv_1_5 = nn.Sequential(
                nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (1,5), padding = (0,2), dilation=1, groups=in_channels//4),
                nn.Conv2d(in_channels//4, in_channels//4, kernel_size = (5,1), padding = (2,0), dilation=1, groups=in_channels//4)
        )  
        self.max_3_1 = nn.Sequential(
                nn.MaxPool2d(3, 1, 1),
                nn.Conv2d(in_channels//4, in_channels//4, kernel_size = 1, padding = 0, dilation=1, groups=in_channels//4)
        )
    def forward(self, x):
        # print("x = ", x)
        # print("self.in_channels = ", self.in_channels)
        a, b, c, d= torch.split(x, self.in_channels//4, dim = 1)
        out_1 = self.conv_1(a)    
        out_2 = self.conv_1_3(b)
        out_3 = self.conv_1_5(c)
        out_4 = self.max_3_1(d)
        out_5 = torch.cat((out_1, out_2, out_3, out_4), 1)
        # out_5 = torch.cat((out_1, out_2), 1)
        out_5 = out_5 + self.gap(x) 
        return out_5
    
# class wf_inception_module(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.in_channels = in_channels
#         self.conv_1 = nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=1, padding=0, dilation=1, groups=in_channels // 4)
#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv_1_3 = nn.Sequential(
#             nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=(1, 3), padding=(0, 1), dilation=1, groups=in_channels // 4),
#             nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=(3, 1), padding=(1, 0), dilation=1, groups=in_channels // 4)
#         )
#         self.conv_1_5 = nn.Sequential(
#             nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=(1, 5), padding=(0, 2), dilation=1, groups=in_channels // 4),
#             nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=(5, 1), padding=(2, 0), dilation=1, groups=in_channels // 4)
#         )
#         self.max_3_1 = nn.Sequential(
#             nn.MaxPool2d(3, 1, 1),
#             nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=1, padding=0, dilation=1, groups=in_channels // 4)
#         )

#     def forward(self, x):
#         # Calculate the size of each split
#         split_size = self.in_channels // 4
#         # List to hold the results
#         results = []

#         # Perform the split and store results
#         for i in range(4):
#             start = i * split_size
#             end = start + split_size
#             if i == 3:
#                 end = self.in_channels  # Handle the last part
#             results.append(x[:, start:end])

#         # Unpack the results
#         a, b, c, d = results

#         # Example adjustment if sizes don't match
#         if a.size(1) != b.size(1):
#             adjust = nn.Conv2d(a.size(1), b.size(1), kernel_size=1, padding=0)
#             a = adjust(a)

#         out_1 = self.conv_1(a)
#         out_2 = self.conv_1_3(b)
#         out_3 = self.conv_1_5(c)
#         out_4 = self.max_3_1(d)
#         out_5 = torch.cat((out_1, out_2, out_3, out_4), 1)
#         out_5 = out_5 + self.gap(x)
#         # print(x.shape)
#         return out_5




    
class STEM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # print("in_channels = ", in_channels)
        self.stem = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_channels, eps=.001),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm2d(out_channels, eps=.001),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm2d(out_channels, eps=.001),
        nn.ReLU(inplace=True),
        # nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
        # nn.           
    )

    def forward(self, x):
        # print("self.stem = ", self.stem)
        x = self.stem(x)
        return x
    

# backbone:
#   # [from, number, module, args]
#   [
#     [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
#     [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
#     [-1, 3, wf_inception_module, [64]],

#     [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
#     [-1, 1, STEM, [128,128]],

#     [-1, 1, Conv, [512, 3, 1]], # 5-P4/16 
#     [-1, 1, STEM, [256,256]],

#     [-1, 1, Conv, [1024, 3, 1]], # 7-P5/32
#     [-1, 1, STEM, [512,512]],
#     [-1, 1, SPPF, [1024, 5]], # 9
    
#   ]

# def autopad1(kernel_size, padding=None):
#     # Pad to 'same'
#     if padding is None:
#         padding = kernel_size // 2 if isinstance(kernel_size, int) else [k // 2 for k in kernel_size]
#     return padding

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        
        if padding == 'same':
            # Calculate padding to achieve 'same' padding behavior
            padding = (kernel_size - 1) // 2
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                                   padding=0, dilation=dilation, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# =++++++++++++++++++++++++++++++++++++++++++
    
# BiFPN
    
# from torchvision.ops.boxes import nms as nms_torch

from efficientnets import EfficientNet as EffNet
from efficientnets.utils import MemoryEfficientSwish, Swish
from efficientnets.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
from torchvision.ops.boxes import nms as nms_torch

def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x


class BiFPN(nn.Module):
    """
    modified by Zylo117
    """

    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True,
                 use_p8=False):
        """

        Args:
            num_channels:
            conv_channels:
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
            onnx_export: if True, use Swish instead of MemoryEfficientSwish
        """
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.use_p8 = use_p8
        print("conv_channels = ",conv_channels)

        # Conv layers
        self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        if use_p8:
            self.conv7_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
            self.conv8_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
        if use_p8:
            self.p7_upsample = nn.Upsample(scale_factor=2, mode='nearest')
            self.p8_downsample = MaxPool2dStaticSamePadding(3, 2)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        self.first_time = first_time
        if self.first_time:
            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )
            if use_p8:
                self.p7_to_p8 = nn.Sequential(
                    MaxPool2dStaticSamePadding(3, 2)
                )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p6_w1_relu = nn.ReLU()
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2_relu = nn.ReLU()
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p7_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """

        # downsample channels using same-padding conv2d to target phase's if not the same
        # judge: same phase as target,
        # if same, pass;
        # elif earlier phase, downsample to target phase's by pooling
        # elif later phase, upsample to target phase's by nearest interpolation

        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = self.p6_w1_relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = self.p7_w2_relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs):
        if self.first_time:
            p3, p4, p5 = inputs

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
            if self.use_p8:
                p8_in = self.p7_to_p8(p7_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

        else:
            if self.use_p8:
                # P3_0, P4_0, P5_0, P6_0, P7_0 and P8_0
                p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            else:
                # P3_0, P4_0, P5_0, P6_0 and P7_0
                p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        if self.use_p8:
            # P8_0 to P8_2

            # Connections for P7_0 and P8_0 to P7_1 respectively
            p7_up = self.conv7_up(self.swish(p7_in + self.p7_upsample(p8_in)))

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_up)))
        else:
            # P7_0 to P7_2

            # Connections for P6_0 and P7_0 to P6_1 respectively
            p6_up = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))

        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_up)))

        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_up)))

        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        if self.use_p8:
            # Connections for P7_0, P7_1 and P6_2 to P7_2 respectively
            p7_out = self.conv7_down(
                self.swish(p7_in + p7_up + self.p7_downsample(p6_out)))

            # Connections for P8_0 and P7_2 to P8_2
            p8_out = self.conv8_down(self.swish(p8_in + self.p8_downsample(p7_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out, p8_out
        else:
            # Connections for P7_0 and P6_2 to P7_2
            p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out

### 加BiFPN模块 ###
# class BiFPN_Concat2(nn.Module):
#     # def __init__(self, dimension=1):
#     #     super(BiFPN_Concat2, self).__init__()
#     #     self.d = dimension
#     #     self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#     #     self.epsilon = 0.0001

#     # def forward(self, x):
#     #     w = self.w
#     #     weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
#     #     # Fast normalized fusion
#     #     x = [weight[0] * x[0], weight[1] * x[1]]
#     #     return torch.cat(x, self.d)
#     # Concatenate a list of tensors along dimension
#     def __init__(self, c1, c2):
#         super(Concat, self).__init__()
#         # self.relu = nn.ReLU()
#         self.w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#         self.epsilon = 0.0001
#         self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
#         self.swish = MemoryEfficientSwish()

#     def forward(self, x):
#         outs = self._forward(x)
#         return outs

#     def _forward(self, x): # intermediate result
#         if len(x) == 2:
#             # w = self.relu(self.w1)
#             w = self.w1
#             weight = w / (torch.sum(w, dim=0) + self.epsilon)
#             x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1]))
#         elif len(x) == 3: # final result
#             # w = self.relu(self.w2)
#             w = self.w2
#             weight = w / (torch.sum(w, dim=0) + self.epsilon)
#             x = self.conv(self.swish(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))

# # 三个分支concat操作
# class BiFPN_Concat3(nn.Module):
#     def __init__(self, dimension=1):
#         super(BiFPN_Concat3, self).__init__()
#         self.d = dimension
#         # 设置可学习参数, nn.Parameter的作用是: 将一个不可训练的类型Tensor转换成可以训练的类型parameter
#         # 并将这个parameter绑定到这个module里面(net.parameter()会返回这个绑定的parameter)
#         # 从而在参数优化的时候可以进行优化
#         self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#         self.epsilon = 0.0001

#     def forward(self, x):
#         w = self.w
#         weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
#         # Fast normalized fusion
#         x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
#         return torch.cat(x, self.d)
        
# ===========================================
        
def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))

    return result


class RepVGGB(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGB, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.SiLU()

        self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0  # mg
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def forward(self, inputs):
        if self.deploy:
            return self.nonlinearity(self.rbr_dense(inputs))
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


class QARepVGGB(RepVGGB):
    """
    RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://arxiv.org/abs/2212.01593
    """

    def __init__(self, in_channels, dim, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(QARepVGGB, self).__init__(in_channels, dim, kernel_size, stride, padding, dilation, groups,
                                        padding_mode, deploy, use_se)
        if not deploy:
            self.bn = nn.BatchNorm2d(dim)
            self.rbr_1x1 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=stride, groups=groups, bias=False)
            self.rbr_identity = nn.Identity() if dim == in_channels and stride == 1 else None
        self._id_tensor = None

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):  # mg
            return self.nonlinearity(self.bn(self.se(self.rbr_reparam(inputs))))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.bn(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(self.rbr_1x1.weight)
        bias = bias3x3  # @CSDN芒果汁没有芒果

        if self.rbr_identity is not None:
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
            for i in range(self.in_channels):
                kernel_value[i, i % input_dim, 1, 1] = 1
            id_tensor = torch.from_numpy(kernel_value).to(self.rbr_1x1.weight.device)
            kernel = kernel + id_tensor
        return kernel, bias

    def deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

    def _fuse_extra_bn_tensor(self, kernel, bias, branch):
        assert isinstance(branch, nn.BatchNorm2d)
        running_mean = branch.running_mean - bias  # remove bias
        running_var = branch.running_var
        gamma = branch.weight
        beta = branch.bias
        eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class QARepNeXt(nn.Module):
    '''
        QARepNeXt is a stage block with qarep-style basic block
    '''

    def __init__(self, in_channels, out_channels, n=1, isTrue=None):
        super().__init__()
        self.conv1 = QARepVGGB(in_channels, out_channels)  # mg
        self.block = nn.Sequential(*(QARepVGGB(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        # print("QARepNeXt = ", x.shape)
        return x
    
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
    
import torch.nn.functional as F
class LayerNorms(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class FocalNextBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.dwconv_3 = nn.Conv2d(dim, dim, kernel_size=7, padding='same', groups=dim, dilation=3)  # depthwise conv

        self.norm = LayerNorms(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1 = x + self.drop_path2(input)
        x = self.dwconv_3(x1)
        x = x + self.drop_path3(x1)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class FocalNext(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(FocalNextBlock(c_) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
    
class Transpose(nn.Module):
    '''Normal Transpose, default for upsampling'''

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True
        )

    def forward(self, x):
        # print("x.shape upsample_transpose = ", x.shape)
        return self.upsample_transpose(x)
    
#--------------------NADH  改进2 --------------------------

class NADH(nn.Module):
    def __init__(self, ch=256, nc=80, anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        #self.cls_1 = Conv(ch, 256, 3, 1, 1)
        #self.merge = Conv(ch, 256, 1, 1)



        self.cls_convs1 = GhostConv(ch, 256, 3, 1, 1)
        self.cls_convs2 = DWConv(256, 128, 3, 1, 1)
        self.cls_convs3 = DWConv(128, 64, 3, 1, 1)

        self.reg_convs1 = GhostConv(ch, 256, 3, 1, 1)
        self.reg_convs2 = GhostConv(256, 128, 3, 1, 1)
        self.reg_convs3 = GhostConv(128, 64, 3, 1, 1)

        self.obj_convs1 = GhostConv(ch, 256, 3, 1, 1)
        self.obj_convs2 = DWConv(256, 128, 3, 1, 1)
        self.obj_convs3 = DWConv(128, 64, 3, 1, 1)

        # self.cls_convs1 = Conv(ch, 256, 3, 1, 1)
        # # self.cls_convs2 = DWConv(256, 128, 3, 1, 1)
        # # self.cls_convs3 = SimConv(128, 64, 3, 1, 1)
        #
        # self.reg_convs1 = Conv(ch, 256, 3, 1, 1)
        # self.reg_convs2 = Conv(256, 128, 3, 1, 1)
        # self.reg_convs3 = Conv(128, 64, 3, 1, 1)
        #
        # self.obj_convs1 = Conv(ch, 256, 3, 1, 1)
        # self.obj_convs2 = Conv(256, 128, 3, 1, 1)
        # self.obj_convs3 = Conv(128, 64, 3, 1, 1)


        # self.cls_convs1 = Conv(ch, 256, 3, 1, 1)
        # self.cls_convs2 = DWConv(256, 128, 3, 1, 1)
        # self.cls_convs3 = DWConv(128, 64, 3, 1, 1)
        #
        #
        # self.reg_convs1 = Conv(ch, 256, 3, 1, 1)
        # self.reg_convs2 = Conv(256, 128, 3, 1, 1)
        # self.reg_convs3 = Conv(128, 64, 3, 1, 1)
        #
        #
        # self.obj_convs1 = Conv(ch, 256, 3, 1, 1)
        # self.obj_convs2 = SimConv(256, 128, 3, 1, 1)
        # self.obj_convs3 = SimConv(128, 64, 3, 1, 1)




        self.cls_preds = nn.Conv2d(64, self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(64, 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(64, 1 * self.na, 1)

    def forward(self, x):
       # x = self.merge(x)

        # q1 = self.cls_convs1(x)
        # x11 = self.cls_preds(q1)

        x1 = self.cls_convs3(self.cls_convs2(self.cls_convs1(x)))
        x11 = self.cls_preds(x1)

        x2 = self.reg_convs3(self.reg_convs2(self.reg_convs1(x)))
        x21 = self.reg_preds(x2)

        # x3 = self.obj_convs1(x)
        # x31 = self.obj_preds(x3)

        x3 = self.obj_convs3(self.obj_convs2(self.obj_convs1(x)))
        x31 = self.obj_preds(x3)


        out = torch.cat([x21, x31, x11], 1)
        return out



class NADHDetect(nn.Module):
    stride = None
    onnx_dynamic = False
    export = False

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(NADH(x, nc, anchors) for x in ch)
        self.inplace = inplace

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
            yv, xv = torch.meshgrid(y, x, indexing='ij')
        else:
            yv, xv = torch.meshgrid(y, x)
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
    
# ====================================================
    
class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)
    
class C3CCFF(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # print("CABottleneck c_ = ", c_)
        # print("CABottleneck c1 = ", c1)
        # print("CABottleneck c2 = ", c2)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        # print("c3 = ", x.shape)
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class RepVGGBlock(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # print("CABottleneck c_ = ", c_)
        # print("CABottleneck c1 = ", c1)
        # print("CABottleneck c2 = ", c2)
        self.cv1 = Conv(c1, c_, 1, 2)
        self.cv2 = Conv(c1, c_, 3, 2)

    def forward(self, x):
        # print("c3 = ", x.shape)
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv1(x) + self.cv2(x)
    

class Inference(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes C3 module with options for channel count, bottleneck repetition, shortcut usage, group
        convolutions, and expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # print("CABottleneck c_ = ", c_)
        # print("c1 = ", c1)
        # print("c2 = ", c2)
        self.cv1 = Conv(c1, c_, 3, 2)
        self.cv2 = Conv(c1, c_, 1, 2)

    def forward(self, x):
        # print("Inference = ", x.shape)
        """Performs forward propagation using concatenated outputs from two convolutions and a Bottleneck sequence."""
        return self.cv1(x) + self.cv2(x)

class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        print("x1 = ", x.shape)
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)