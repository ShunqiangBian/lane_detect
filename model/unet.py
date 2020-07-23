#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.network import ResNet101v2
from model.module import Block

class UNetConvBlock(nn.Module):
    '''下采样
    :param in_chans：输入的通道数
    :param out_chans：输出的通道数
    :param padding：填充的大小
    :param batch_norm：bn层
    '''

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

class UNetUpBlock(nn.Module):
    '''上采样
    '''
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

class UNet(nn.Module):
    def __init__(
            self,
            in_channels=3,  # 输入的channel数
            n_classes=2,  # 最终有多少个分类
            depth=5,  # 网络的深度
            wf=6,  # 第一层的个数，2的几次方
            padding=1,
            batch_norm=False,
            up_mode='upconv',
    ):
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels

        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        block = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                block.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, block[-i - 1])
        return self.last(x)


class ResNetUNet(nn.Module):
    def __init__(self,args
    ):
        super(ResNetUNet, self).__init__()
        # 网络参数要跟deeplabv3p一样的参数，是同一个config
        self.n_classes = 8
        self.padding = 1
        self.up_mode = 'upconv'
        assert self.up_mode in ('upconv', 'upsample')
        # encode改成resnet101v2
        self.encode = ResNet101v2()
        # 上一层的给出的就是2048
        prev_channels = 2048
        self.up_path = nn.ModuleList()
        self.batch_norm = True

        for i in range(3):
            self.up_path.append(
                UNetUpBlock(prev_channels, prev_channels // 2, self.up_mode, self.padding, self.batch_norm)
            )
            prev_channels //= 2

        self.cls_conv_block1 = Block(prev_channels, 32)
        self.cls_conv_block2 = Block(32, 16)
        self.last = nn.Conv2d(16, self.n_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # blocks就是f2到f5
        input_size = x.size()[2:]
        blocks = self.encode(x)
        # 最后一个作为上采样的输入
        x = blocks[-1]

        # 对up_path进行for循环
        for i, up in enumerate(self.up_path):
            # 将三个上采样的blocks都执行一遍，输入是x，输出是网络对应的feature
            x = up(x, blocks[-i - 2])

        # 进行上采样，上采样成输入的尺寸，align_corners的意思是rensize的时候，边缘是不是跟原图对齐
        x = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)(x)
        x = self.cls_conv_block1(x)
        x = self.cls_conv_block2(x)
        x = self.last(x)
        return x
