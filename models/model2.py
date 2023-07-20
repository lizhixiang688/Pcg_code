#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model2.py
# @Author: Lzx
# @Date  : 2021/12/29
# @Desc  :

import torch
import torch.nn as nn
from models.quaternion_layers import QuaternionConv, QuaternionLinear
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, num_class):
        super(QNet, self).__init__()

        self.conv = nn.Conv1d(1, 12, kernel_size=1, stride=1, padding=0)

        self.conv1 = QuaternionConv(12, 64, kernel_size=3, stride=1, padding=1, operation='convolution1d')
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = QuaternionConv(64, 64, kernel_size=3, stride=1, padding=1, operation='convolution1d')
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv3 = QuaternionConv(64, 128, kernel_size=3, stride=1, padding=1, operation='convolution1d')
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = QuaternionConv(128, 128, kernel_size=3, stride=1, padding=1, operation='convolution1d')
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.conv5 = QuaternionConv(128, 256, kernel_size=3, stride=1, padding=1, operation='convolution1d')
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = QuaternionConv(256, 256, kernel_size=3, stride=1, padding=1, operation='convolution1d')
        self.bn6 = nn.BatchNorm1d(256)
        self.conv7 = QuaternionConv(256, 256, kernel_size=3, stride=1, padding=1, operation='convolution1d')
        self.bn7 = nn.BatchNorm1d(256)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3, stride=3)

        self.branch1 = nn.Sequential(
            QuaternionConv(256, 512, kernel_size=3, stride=1, padding=1, operation='convolution1d'),
            nn.BatchNorm1d(512),
            QuaternionConv(512, 512, kernel_size=3, stride=1, padding=1, operation='convolution1d'),
            nn.BatchNorm1d(512),
            QuaternionConv(512, 512, kernel_size=3, stride=1, padding=1, operation='convolution1d'),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(kernel_size=3, stride=3),

            QuaternionConv(512, 512, kernel_size=3, stride=1, padding=1, operation='convolution1d'),
            nn.BatchNorm1d(512),
            QuaternionConv(512, 256, kernel_size=3, stride=1, padding=1, operation='convolution1d'),
            nn.BatchNorm1d(256),
            QuaternionConv(256, 128, kernel_size=3, stride=1, padding=1, operation='convolution1d'),
            nn.BatchNorm1d(128)
        )

        self.branch2 = nn.Sequential(
            QuaternionConv(256, 512, kernel_size=3, stride=1, padding=3, operation='convolution1d', dilatation=3),
            nn.BatchNorm1d(512),
            QuaternionConv(512, 512, kernel_size=3, stride=1, padding=3, operation='convolution1d', dilatation=3),
            nn.BatchNorm1d(512),
            QuaternionConv(512, 512, kernel_size=3, stride=1, padding=3, operation='convolution1d', dilatation=3),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(kernel_size=3, stride=3),

            QuaternionConv(512, 512, kernel_size=3, stride=1, padding=3, operation='convolution1d', dilatation=3),
            nn.BatchNorm1d(512),
            QuaternionConv(512, 256, kernel_size=3, stride=1, padding=3, operation='convolution1d', dilatation=3),
            nn.BatchNorm1d(256),
            QuaternionConv(256, 128, kernel_size=3, stride=1, padding=3, operation='convolution1d', dilatation=3),
            nn.BatchNorm1d(128)
        )
        self.se = SELayer(128, 16)

        self.avg = nn.AdaptiveAvgPool1d(output_size=1)

        # self.lin_out1_1 = QuaternionLinear(128, 32)
        # self.lin_out1_2 = nn.Linear(32, num_class)
        #
        # self.lin_out2_1 = QuaternionLinear(128, 32)
        # self.lin_out2_2 = nn.Linear(32, num_class)

        self.lin1 = QuaternionLinear(128, 64)
        self.lin2 = QuaternionLinear(64, 32)
        self.lin3 = nn.Linear(32, num_class)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):

        x = self.conv(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.maxpool2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.maxpool3(x)

        out1 = self.branch1(x)   # out1=>[b, 128, 37]
        out2 = self.branch2(x)   # out2=>[b, 128, 37]

        x = self.se(out1 + out2)
        x = self.avg(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x = F.relu(self.lin2(x))
        x = self.drop(x)
        x = self.lin3(x)

        # out1 = self.avg(out1)
        # out1 = out1.view(out1.size(0), -1)
        # out1 = F.relu(self.lin_out1_1(out1))
        # out1 = self.drop(out1)
        # out1 = self.lin_out1_2(out1)
        #
        # out2 = self.avg(out2)
        # out2 = out2.view(out2.size(0), -1)
        # out2 = F.relu(self.lin_out1_1(out2))
        # out2 = self.drop(out2)
        # out2 = self.lin_out1_2(out2)

        return x  # 这里我不是很清楚double-loss的处理方法 需不需要返回两个分支的output


class SELayer(nn.Module):    # 这个是四元数的注意力模块
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
    model = QNet(2)
    x = model(x)
    print(x.shape)

