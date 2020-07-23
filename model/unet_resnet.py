# -*- coding: utf-8 -*-
"""使用 Resnet 作为 encoder 网络的 Unet 模型
"""
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 卷积操作
    :param in_planes: 输入通道数
    :param out_planes: 输出通道数
    :param stride: 步长
    :param padding: 填充大小
    :return Conv2d object: 3x3 卷积层
    :note: 如果卷积层之后是BN层，那么可以不用偏置参数，可以节省内存
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积操作
    :param in_planes: 输入通道数
    :param out_planes: 输出通道数
    :param stride: 步长
    :return Conv2d object: 1x1 卷积层:
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


"""残差网络的basicblock
"""


class BasicBlock(nn.Module):
    expansion = 1  # 经过Block之后channel的变化量

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        # downsample: 调整维度一致之后才能相加
        # norm_layer：batch normalization layer
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 如果bn层没有自定义，就使用标准的bn层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 保存x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # downsample调整x的维度，F(x)+x一致才能相加

        out += identity
        out = self.relu(out)  # 先相加再激活

        return out


"""残差网络的bottleneck
"""


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)  # 输入的channel数：planes * self.expansion
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64

        # conv1 in ppt figure
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """stage/layer 生成函数
        :param block: 选择 Bottleneck 还是 Basicblock
        :param planes: 输入输出通道数
        :param blocks: 残差 blocks 的数量
        :param stride: 步长
        :return:
        """
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 需要调整维度
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),  # 同时调整spatial(H x W))和channel两个方向
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))  # 第一个block单独处理
        self.inplanes = planes * block.expansion  # 记录layerN的channel变化，具体请看ppt resnet表格
        for _ in range(1, blocks):  # 从1开始循环，因为第一个模块前面已经单独处理
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)  # 使用Sequential层组合blocks，形成stage。如果layers=[2,3,4]，那么*layers=？

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1, x2, x3, x4, x5


# def _resnet(block, layers):
#     x1,x2, x3, x4, x5 = ResNet(block, layers)
#     return x1,x2, x3, x4, x5


def resnet101():
    """ResNet-101 model from
    :param pretrained: If True, returns a model pre-trained on ImageNet
    :param progress: If True, displays a progress bar of the download to stderr
    :param kwargs: 自定义参数
    :return resnet object: 返回残差网络模型
    """
    return ResNet(Bottleneck, [3, 4, 6, 3])


"""ResnetUnet 的主体网络
"""


class ResNetUNet(nn.Module):
    def __init__(
            self
    ):
        super(ResNetUNet, self).__init__()
        # 网络参数要跟deeplabv3p一样的参数，是同一个config
        self.n_classes = 8
        self.padding = 1
        self.up_mode = 'upconv'
        assert self.up_mode in ('upconv', 'upsample')
        # encode改成resnet101
        self.encode = resnet101()
        # 上一层的给出的就是2048
        prev_channels = 2048
        self.up_path = nn.ModuleList()
        self.batch_norm = True
        for i in range(4):
            self.up_path.append(
                UNetUpBlock(prev_channels, prev_channels // 2, self.up_mode, self.padding, self.batch_norm)
            )
        self.last = nn.Conv2d(prev_channels // 2, self.n_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        blocks = self.encode(x)
        # 最后一个作为上采样的输入
        x = blocks[-1]
        # 对up_path进行for循环
        for i, up in enumerate(self.up_path):
            # 将四个上采样的blocks都执行一遍，输入是x，输出是网络对应的feature
            x = up(x, blocks[-i - 1])
        x = self.last(x)
        return x


'''Unet 上采样 Block Class
'''


class UNetUpBlock(nn.Module):
    def __init__(self, in_chans, out_chans, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_chans, out_chans, kernel_size=1), )
        self.conv_block = UNetConvBlock(in_chans, out_chans, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        deff_y = (layer_height - target_size[0]) // 2
        deff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, deff_y:(deff_y + target_size[0]), deff_x:(deff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out


'''Unet basic block class
'''


class UNetConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        block.append(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_chans))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out
