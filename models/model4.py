
# 这是效果最好的模型结构
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
        self.conv1 = nn.Conv1d(39, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(39, 64, kernel_size=1, stride=1)  # 用于提取通道关联信息
        self.bn2 = nn.BatchNorm1d(64)
        # self.dw = Normal_DWBlock(64, 3, dynamic=True)

        # self.lin = nn.Sequential(
        #     nn.Linear(1495, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 128),
        #     nn.ReLU(inplace=True),
        # )

        self.blk1 = Qblock(64, 16)
        self.blk2 = Qblock(64, 16)
        self.blk3 = Qblock(64, 32)
        self.blk4 = Qblock(128, 32)
        self.blk5 = Qblock(128, 32)

        self.conv3 = QuaternionConv(128, 64, kernel_size=3, stride=1, padding=1, operation='convolution1d')  # 降低维度
        self.bn3 = nn.BatchNorm1d(64)

        self.avg = nn.AdaptiveAvgPool1d(output_size=1)

        self.lin1 = QuaternionLinear(64, 32)
        self.lin2 = QuaternionLinear(32, 16)
        self.lin3 = QuaternionLinear(16, 8)
        self.lin4 = nn.Linear(8, num_class)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        res = x
        res = self.bn2(self.conv2(res))
        # res = self.dw(res)

        # raw = x[:, 1:5, :]
        # raw = raw.view(raw.size(0), -1)
        # raw = self.lin(raw)

        x = self.bn1(self.conv1(x))

        x1 = self.blk1(x)
        x2 = self.blk2(x1)

        x3 = x1 + x2
        x3 = self.blk3(x3)
        x4 = self.blk4(x3)

        x5 = x3 + x4
        x = self.blk5(x5)

        # x = torch.cat([x, res], dim=1)
        x = self.bn3(self.conv3(x))  # [128-->64]
        x = res + x  # [b, 64, 299]

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        # x = torch.cat([x, raw], dim=1)

        x = F.relu(self.lin1(x))
        x = self.drop(x)
        x = F.relu(self.lin2(x))
        x = self.drop(x)
        x = F.relu(self.lin3(x))
        x = self.drop(x)
        x = self.lin4(x)

        return x


class Qblock(nn.Module):  # 这个就相当于一个带注意力机制的残差块
    def __init__(self, ch_in, ch_out, stride=1):
        super(Qblock, self).__init__()

        self.conv1 = QuaternionConv(ch_in, ch_out, kernel_size=1, stride=1, operation='convolution1d')
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.conv2 = QuaternionConv(ch_out, ch_out, kernel_size=3, stride=1, padding=1, operation='convolution1d')
        self.bn2 = nn.BatchNorm1d(ch_out)
        self.conv3 = QuaternionConv(ch_out, ch_out * 4, kernel_size=1, stride=1, operation='convolution1d')
        self.bn3 = nn.BatchNorm1d(ch_out * 4)
        self.dwblock = DWBlock(ch_out * 4, 3, dynamic=True)

        self.cam = CAM(channel=ch_out * 4, reduction=16)  # 这里的注意力机制可以修改

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
        x = self.dwblock(x)
        x = self.cam(x)
        # x = self.sam(x)

        res = self.extra(res)
        x = F.relu(x + res)

        return x


class CAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.conv1 = nn.Sequential(
            QuaternionConv(channel, channel // reduction, kernel_size=1, stride=1, operation='convolution1d'),  # 注意这里如果reduction=16的话，channel必须大于64.
                                                                                                                # 这里channel//reduction >4
            nn.ReLU(True),
            QuaternionConv(channel // reduction, channel, kernel_size=1, stride=1, operation='convolution1d'),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y1 = self.avg_pool(x)  # [16, 32, 100]=>[16, 32, 1]
        y1 = self.conv1(y1)

        y2 = self.max_pool(x)
        y2 = self.conv1(y2)

        y = self.sigmoid(y1 + y2)

        return x * y  # 这里的y不需要扩展，因为pytorch有broadcast机制，自动补全


if __name__ == '__main__':
    device = torch.device('cuda')

    a = torch.randn(4, 39, 299)
    model = QCnn(2)
    a = model(a)
    print(a.shape)



