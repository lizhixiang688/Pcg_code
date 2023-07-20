#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : PcgNet.py
# @Author: Lzx
# @Date  : 2022/1/14
# @Desc  :

import torch
import torch.nn as nn
from models.quaternion_layers import QuaternionConv, QuaternionLinear
import torch.nn.functional as F


class QCnn(nn.Module):
    def __init__(self, num_class):
        super(QCnn, self).__init__()

        self.num_class = num_class
        self.conv1 = nn.Conv1d(1, 4, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(4)
        self.conv2 = QuaternionConv(4, 16, kernel_size=5, stride=1, padding=2, operation='convolution1d')
        self.bn2 = nn.BatchNorm1d(16)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.blk1 = Qblock(16, 16)
        self.blk2 = Qblock(64, 32)
        self.blk3 = Qblock(128, 64)
        self.blk4 = Qblock(256, 128)

        self.avg = nn.AdaptiveAvgPool1d(output_size=1)
        self.lin1 = QuaternionLinear(512, 128)
        self.lin2 = QuaternionLinear(128, 32)
        self.lin3 = nn.Linear(32, num_class)
        self.drop = nn.Dropout(0.5)



    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x = F.relu(self.lin2(x))
        x = self.drop(x)
        x = self.lin3(x)

        return x


class Qblock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(Qblock, self).__init__()

        self.conv1 = QuaternionConv(ch_in, ch_out, kernel_size=1, stride=1, operation='convolution1d')
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.conv2 = QuaternionConv(ch_out, ch_out, kernel_size=3, stride=1, padding=1, operation='convolution1d')
        self.bn2 = nn.BatchNorm1d(ch_out)
        self.conv3 = QuaternionConv(ch_out, ch_out * 4, kernel_size=1, stride=1, operation='convolution1d')
        self.bn3 = nn.BatchNorm1d(ch_out * 4)

        self.se = SELayer(channel=ch_out * 4, reduction=16)
        self.extra = nn.Sequential()   # 这个是如果维度不一样就用1*1卷积来使维度相同
        if ch_in != ch_out * 4:
            self.extra = nn.Sequential(
                QuaternionConv(ch_in, ch_out * 4, kernel_size=1, stride=1, operation='convolution1d'),
                nn.BatchNorm1d(ch_out * 4)
            )

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se(x)

        res = self.extra(res)
        x = F.relu(x + res)

        return x


class SELayer(nn.Module):  # 这个是四元数的注意力模块
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 比如：x[16, 32, 100] => [16, 32, 1]
        self.fc = nn.Sequential(
            QuaternionLinear(channel, channel // reduction),  # [16, 32]=>[16, 2]
            nn.ReLU(inplace=True),
            QuaternionLinear(channel // reduction, channel),  # [16, 2]=>[16, 32]
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()  # b:16(batch),c:32(channel)
        y = self.avg_pool(x).view(b, c)  # [16, 32, 1]=>[16, 32]
        y = self.fc(y).view(b, c, 1)  # [16, 32]=>[16, 32, 1]
        return x * y.expand_as(x)  # [16, 32, 100]


if __name__ == '__main__':
    x = torch.randn(8, 1, 6000)
    model = QCnn(2)
    x = model(x)
    print(x.shape)
