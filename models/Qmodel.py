#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Qmodel.py
# @Author: Lzx
# @Date  : 2021/12/24
# @Desc  :

import torch
import torch.nn as nn
from models.quaternion_layers import QuaternionConv, QuaternionLinear
import torch.nn.functional as F
from models.recurrent_models import QRNN, QLSTM


class QCnn(nn.Module):
    def __init__(self, num_class):
        super(QCnn, self).__init__()

        self.num_class = num_class
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.blk1 = Qblock(32, 16)
        self.blk2 = Qblock(64, 16)
        self.blk3 = Qblock(64, 32)
        self.blk4 = Qblock(128, 32)
        self.blk5 = Qblock(128, 64)
        self.blk6 = Qblock(256, 64)

        self.avg = nn.AdaptiveAvgPool1d(output_size=1)
        self.max = nn.AdaptiveMaxPool1d(output_size=1)

        self.lin1 = QuaternionLinear(256, 128)
        self.lin2 = QuaternionLinear(128, 32)
        self.lin3 = nn.Linear(32, num_class)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.maxpool(x)

        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = self.blk5(x)
        x = self.blk6(x)

        avg = self.avg(x)
        max = self.max(x)
        x = avg + max

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


class Cnn(nn.Module):   # 这个是和Qcnn一样的结构，只是没有用四元数层而已
    def __init__(self, num_class):
        super(Cnn, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(4)

        self.conv2 = nn.Conv1d(4, 16, kernel_size=15, stride=2, padding=7)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(16, 64, kernel_size=15, stride=2, padding=7)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7)
        self.bn5 = nn.BatchNorm1d(256)

        # self.conv6 = nn.Conv1d(256, 256, kernel_size=1, stride=1, padding=0)
        # self.bn6 = nn.BatchNorm1d(256)

        self.conv7 = nn.Conv1d(256, 128, kernel_size=9, stride=1, padding=4)
        self.bn7 = nn.BatchNorm1d(128)
        self.conv8 = nn.Conv1d(128, 64, kernel_size=9, stride=1, padding=4)
        self.bn8 = nn.BatchNorm1d(64)
        # self.conv9 = nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=0)
        # self.bn9 = nn.BatchNorm1d(32)

        self.avg = nn.AdaptiveAvgPool1d(output_size=1)

        self.linear1 = nn.Linear(64, 16)
        self.linear2 = nn.Linear(16, num_class)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        # x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        # x = F.relu(self.bn9(self.conv9(x)))

        x = self.avg(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class QRnn(nn.Module):  # 注意这里是小写的Rnn
    def __init__(self, num_class):
        super(QRnn, self).__init__()

        self.rnn1 = nn.RNN(input_size=1, hidden_size=4, num_layers=1, batch_first=True)
        self.Qrnn1 = QRNN(4, 16, CUDA=True)
        self.Qrnn2 = QRNN(16, 32, CUDA=True)
        self.Qrnn3 = QRNN(32, 64, CUDA=True)
        self.Qrnn4 = QRNN(64, 128, CUDA=True)
        self.Qrnn5 = QRNN(128, 256, CUDA=True)

        self.avg = nn.AdaptiveAvgPool1d(output_size=1)

        self.lin1 = nn.Linear(3000, 1024)
        self.lin2 = nn.Linear(1024, 128)
        self.lin3 = nn.Linear(128, num_class)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x, hiden = self.rnn1(x)

        x = x.permute(1, 0, 2)  # 这里由于QRNN需要的输入是 [seq, batch, feat_size]
                               # x[512, 6000, 4]=>[6000, 512, 4]
        x = self.Qrnn1(x)
        x = self.Qrnn2(x)
        x = self.Qrnn3(x)
        x = self.Qrnn4(x)
        x = self.Qrnn5(x)
        x = x.permute(1, 0, 2)  # 在rnn之后，又要把batch和sequence换回来

        x = self.avg(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x = F.relu(self.lin2(x))
        x = self.drop(x)
        x = F.relu(self.lin3(x))


        return x


class Qlstm(nn.Module):
    def __init__(self, num_class):
        super(Qlstm, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=4, num_layers=1)

        self.Qlstm1 = QLSTM(4, 16, CUDA=False)
        self.Qlstm2 = QLSTM(16, 32, CUDA=False)
        self.Qlstm3 = QLSTM(32, 64, CUDA=False)
        self.Qlstm4 = QLSTM(64, 128, CUDA=False)
        self.Qlstm5 = QLSTM(128, 256, CUDA=False)

        self.avg = nn.AdaptiveAvgPool1d(output_size=1)

        self.lin1 = nn.Linear(6000, 1024)
        self.lin2 = nn.Linear(1024, 128)
        self.lin3 = nn.Linear(128, num_class)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # 这里由于QLSTM需要的输入是 [seq, batch, feat_size]
        # x[512, 6000, 4]=>[6000, 512, 4]

        x, (h, c) = self.lstm1(x)

        x = self.Qlstm1(x)
        x = self.Qlstm2(x)
        x = self.Qlstm3(x)
        x = self.Qlstm4(x)
        x = self.Qlstm5(x)
        x = x.permute(1, 0, 2)  # 在lstm之后，又要把batch和sequence换回来  # 变成[batch, sequence, hidden_size]

        x = self.avg(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x = F.relu(self.lin2(x))
        x = self.drop(x)
        x = F.relu(self.lin3(x))

        return x


if __name__ == '__main__':
    device = torch.device('cuda')
    x = torch.randn(4, 1, 6000)
    model = QCnn(2)
    x = model(x)
    print(x.shape)

