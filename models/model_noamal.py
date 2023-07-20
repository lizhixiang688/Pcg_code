
import torch
import torch.nn as nn
from models.quaternion_layers import QuaternionConv, QuaternionLinear
import torch.nn.functional as F
from models.demo1 import DWBlock
from models.demo2 import Normal_DWBlock


class QCnn(nn.Module):
    def __init__(self, num_class):
        super(QCnn, self).__init__()

        self.num_class = num_class

        self.branch = block(39, 32)

        self.lin = nn.Sequential(
            nn.Linear(299, 128),
            nn.ReLU(inplace=True),
        )

        self.branch_main = Qbranch(in_channel=39, out_channel=64, is_main=True)

        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.avg = nn.AdaptiveAvgPool1d(output_size=1)

        self.lin1 = nn.Linear(256, 64)   # 四元数卷积和普通卷积差不多
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 16)
        self.lin4 = nn.Linear(16, num_class)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        res = x
        res, global_atten = self.branch(res)  # [b, 128, seq]

        x1 = x[:, :1, :]   # 只要能量部分
        x1 = x1.view(x1.size(0), -1)
        x1 = self.lin(x1)   # [b, 128]

        x = self.branch_main(x)  # x [b, 128, 299],

        x = torch.cat([x, res], dim=1)  # [b, 256, 299]

        x = self.bn3(self.conv3(x))  # [b, 128, seq]

        x = self.avg(x)

        x = x.view(x.size(0), -1)  # [b, 128]

        x = torch.cat([x, x1], dim=1)  # [b, 256]

        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x = F.relu(self.lin2(x))
        x = self.drop(x)
        x = F.relu(self.lin3(x))
        x = self.lin4(x)

        return x


class Qbranch(nn.Module):

    def __init__(self, in_channel, out_channel, is_main=False):
        super(Qbranch, self).__init__()

        self.is_main = is_main

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channel)

        self.blk1 = Qblock(out_channel, 16)    # out->64
        self.blk2 = Qblock(64, 16)    # 64->64
        self.blk3 = Qblock(64, 32, same=False)    # 64->128
        self.blk4 = Qblock(128, 32)   # 128->128
        self.blk5 = Qblock(128, 32)   # 128->128

    def forward(self, x):  # x[b, 64, 299]
        x = self.bn1(self.conv1(x))

        x1, attn1 = self.blk1(x, None)
        x2, attn2 = self.blk2(x1, attn1)

        x3 = x1 + x2
        x3, attn3 = self.blk3(x3, attn2)
        x4, attn4 = self.blk4(x3, attn3)

        x5 = x3 + x4
        x5, attn5 = self.blk5(x5, attn4)

        if self.is_main:
            return x5


class Qblock(nn.Module):  # 这个就相当于一个带注意力机制的残差块
    def __init__(self, ch_in, ch_out, stride=1, same=True):
        super(Qblock, self).__init__()

        self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.conv2 = nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(ch_out)
        self.conv3 = nn.Conv1d(ch_out, ch_out * 4, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(ch_out * 4)
        self.dwblock = Normal_DWBlock(ch_out * 4, 3, dynamic=True)  # 实值动态卷积

        self.se = CAM(channel=ch_out * 4, reduction=16, is_same=same)  # 这里的注意力机制可以修改
        self.extra = nn.Sequential()   # 这个是如果维度不一样就用1*1卷积来使维度相同
        if ch_in != ch_out * 4:
            self.extra = nn.Sequential(
                nn.Conv1d(ch_in, ch_out * 4, kernel_size=1, stride=1),
                nn.BatchNorm1d(ch_out * 4)
            )

    def forward(self, x, attn):
        res = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dwblock(x)
        x, attn2 = self.se(x, attn)

        res = self.extra(res)
        x = F.relu(x + res)

        return x, attn2


class block(nn.Module):  # 这个就相当于一个带注意力机制的残差块
    def __init__(self, ch_in, ch_out, stride=1):
        super(block, self).__init__()

        self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.conv2 = nn.Conv1d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(ch_out)
        self.conv3 = nn.Conv1d(ch_out, ch_out * 4, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm1d(ch_out * 4)
        # self.dw = Normal_DWBlock(ch_out * 4, 3, dynamic=True)

        self.se = CAM(channel=ch_out * 4, reduction=16, is_normal=True)  # 这里为正常的模块
        self.extra = nn.Sequential()   # 这个是如果维度不一样就用1*1卷积来使维度相同
        if ch_in != ch_out * 4:
            self.extra = nn.Sequential(
                nn.Conv1d(ch_in, ch_out * 4, kernel_size=1, stride=1),
                nn.BatchNorm1d(ch_out * 4)
            )

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = self.dw(x)
        x, att = self.se(x, None)

        res = self.extra(res)
        x = F.relu(x + res)

        return x, att


class CAM(nn.Module):
    def __init__(self, channel, reduction=16, is_normal=False, is_same=True):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.attn_fc = nn.Sequential()
        if not is_same:
            self.attn_fc = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(inplace=True)
            )

        self.conv2 = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        if is_normal:
            self.conv = nn.Sequential(
                    nn.Conv1d(channel, channel // reduction, kernel_size=1, stride=1),
                    nn.ReLU(True),
                    nn.Conv1d(channel // reduction, channel, kernel_size=1, stride=1)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(channel, channel // reduction, kernel_size=3, stride=1, padding=1),  # 注意这里如果reduction=16的话，channel必须大于64.
                                                                                                                    # 这里channel//reduction >4
                nn.ReLU(True),
                nn.Conv1d(channel // reduction, channel, kernel_size=3, stride=1, padding=1),
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, att):
        y1 = self.avg_pool(x)  # [16, 32, 100]=>[16, 32, 1]
        y1 = self.conv(y1)

        y2 = self.max_pool(x)
        y2 = self.conv(y2)

        y = self.sigmoid(y1 + y2)
        b, c, _ = y.size()
        if att is not None:
            att = att.view(att.size(0), -1)
            att = self.attn_fc(att)    # 这里是对注意力进行维度转换
            att = att.view(b, 1, c)
            y = y.view(b, 1, c)
            y = torch.cat([att, y], dim=1)
            y = self.conv2(y).view(b, c, 1)

        return x * y, y  # 这里的y不需要扩展，因为pytorch有broadcast机制，自动补全


if __name__ == '__main__':
    a = torch.randn(4, 39, 299)
    model = QCnn(2)
    a = model(a)
    print(a.shape)



