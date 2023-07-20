#!D:\Anaconda\envs\test\python
# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 20:44
# @Author  : Lzx
# @File    : demo2.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicDWConv(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, stride=1, padding=1, groups=1, reduction=3):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(dim, dim // reduction, 1, bias=False)
        self.bn = nn.BatchNorm1d(dim // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(dim // reduction, dim * kernel_size, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, seq = x.shape
        y = self.bn(self.conv1(self.pool(x)))
        weight = self.conv2(self.relu(y))  # [b, dim*ker,,1]

        weight = weight.view(b * self.dim, 1, self.kernel_size)  # reshape(1,-1,seq) ==>[1，b*c,seq]
        x = F.conv1d(x.reshape(1, -1, seq), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding,
                     groups=b * self.groups)

        x = x.view(b, c, x.shape[-1])

        return x


class Normal_DWBlock(nn.Module):

    def __init__(self, dim, window_size, dynamic=False, inhomogeneous=False, heads=None):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.dynamic = dynamic
        self.inhomogeneous = inhomogeneous
        self.heads = heads

        # pw-linear
        self.conv0 = nn.Conv1d(dim, dim, 1, bias=False)  # 这里维度不变
        self.bn0 = nn.BatchNorm1d(dim)

        if dynamic and not inhomogeneous:
            self.conv = DynamicDWConv(dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)
        else:
            self.conv = nn.Conv1d(dim, dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim)

        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)

        # pw-linear
        self.conv2 = nn.Conv1d(dim, dim, 1, bias=False)  # 这里也是维度不变
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):  # x [b c h w]

        # x = self.conv0(x)
        # x = self.bn0(x)
        # x = self.relu(x)

        x = self.conv(x)  # 主要就是这里的卷积需要讨论
        x = self.bn(x)
        x = self.relu(x)

        # x = self.conv2(x)
        # x = self.bn2(x)

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
    x = torch.randn(4, 39, 299)
    model = Normal_DWBlock(39, 3, dynamic=True)
    x = model(x)

    print(x.shape)
