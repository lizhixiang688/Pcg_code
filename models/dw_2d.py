#!D:\Anaconda\envs\test\python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/29 15:09
# @Author  : Lzx
# @File    : dw_2d.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicDWConv(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(dim, dim // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim // reduction, dim * kernel_size * kernel_size, 1)
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.conv2(self.relu(self.bn(self.conv1(self.pool(x)))))
        weight = weight.view(b * self.dim, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x


class DWBlock_2d(nn.Module):
    def __init__(self, dim, window_size, dynamic=False, inhomogeneous=False, heads=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.dynamic = dynamic
        self.inhomogeneous = inhomogeneous
        self.heads = heads

        # pw-linear
        self.conv0 = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(dim)

        if dynamic and not inhomogeneous:
            self.conv = DynamicDWConv(dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)
        # if dynamic and inhomogeneous:
        #     print(window_size, heads)
        #     self.conv = IDynamicDWConv(dim, window_size, heads)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)

        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)

        # pw-linear
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):

        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # x = self.conv0(x)
        flops += N * self.dim * self.dim
        # x = self.conv(x)
        if self.dynamic and not self.inhomogeneous:
            flops += (
                        N * self.dim + self.dim * self.dim / 4 + self.dim / 4 * self.dim * self.window_size * self.window_size)
        elif self.dynamic and self.inhomogeneous:
            flops += (
                        N * self.dim * self.dim / 4 + N * self.dim / 4 * self.dim / self.heads * self.window_size * self.window_size)
        flops += N * self.dim * self.window_size * self.window_size
        #  x = self.conv2(x)
        flops += N * self.dim * self.dim
        #  batchnorm + relu
        flops += 8 * self.dim * N
        return flops


if __name__ == '__main__':
    model = DWBlock_2d(16, 3, dynamic=True)
    x = torch.randn(4, 16, 39, 299)
    x = model(x)
    print(x.shape)