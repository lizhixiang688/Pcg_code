import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class Block(nn.Module):
    '''
    Grouped convolution block.
    '''
    expansion = 2  # 膨胀 = 2

    def __init__(self, in_planes, cardinality=16, bottleneck_width=4, stride=1):  # cardinality（基数）= 32
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv1d(in_planes, group_width, kernel_size=7, bias=False, padding=3)
        self.bn1 = nn.BatchNorm1d(group_width)
        self.conv2 = nn.Conv1d(group_width, group_width, kernel_size=11, stride=stride, padding=5, bias=False, groups=cardinality)
        self.bn2 = nn.BatchNorm1d(group_width)
        self.conv3 = nn.Conv1d(group_width, self.expansion*group_width,  kernel_size=7, bias=False, padding=3)
        self.bn3 = nn.BatchNorm1d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=2):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 16
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(num_blocks[0], 2)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.layer4 = self._make_layer(num_blocks[3], 2)

        self.linear = nn.Linear(512, 64)
        self.lin2 = nn.Linear(64, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.layer4(out)
        out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.lin2(out)
        return out

def ResNeXt29_2x64d():
    model = ResNeXt(num_blocks=[3, 3, 3, 3], cardinality=2, bottleneck_width=64)
    return model

def ResNeXt29_4x64d():
    model = ResNeXt(num_blocks=[3, 6, 3], cardinality=4, bottleneck_width=64)
    return model

def ResNeXt29_8x64d():
    model = ResNeXt(num_blocks=[3, 3, 3], cardinality=8, bottleneck_width=64)
    return model


def ResNeXt29_32x4d():
    model = ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=8, bottleneck_width=4)
    return model


if __name__ == '__main__':
    net = ResNeXt29_32x4d()
    x = torch.randn(4, 1, 6000)
    y = net(x)

    print(y.shape)
# test_resnext()
