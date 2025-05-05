import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from .dynamic_conv import Dynamic_conv2d
from functools import partial
from .BottleneckTransformers import ResNet50
# nonlinearity = partial(F.relu, inplace=True)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        X0 = x
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(max_out)
        return X0 * self.sigmoid(x)


class DFIMBlock(nn.Module):
    def __init__(self, channels):
        super(DFIMBlock, self).__init__()

        self.conv_Dyn3 = Dynamic_conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv_Dyn1 = Dynamic_conv2d(channels, channels, kernel_size=1)
        self.gelu3 = nn.GELU()
        self.gelu1 = nn.GELU()
        self.sa = SpatialAttention()

    def forward(self, x):
        x_Dyn3 = self.conv_Dyn3(x)
        G3 = self.gelu3(x_Dyn3)
        x_Dyn1 = self.conv_Dyn1(G3)
        G1 = self.gelu1(x_Dyn1)
        out = x_Dyn3 + G1
        out = self.sa(out)

        return out

class _Get_res(nn.Module):
    def __init__(self, num_classes=1):
        super(_Get_res, self).__init__()

        self.stage1 = Dynamic_conv2d(1024, 256, 3, padding=1)
        # self.stage1_ru = nonlinearity
        self.stage1_ru = nn.GELU()
        self.up1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.stage2 = Dynamic_conv2d(512, 256, 3, padding=1)
        # self.stage2_ru = nonlinearity
        self.stage2_ru = nn.GELU()
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_o = Dynamic_conv2d(256 * 4, 256, 3, padding=1)
        # self.conv1_ru = nonlinearity
        self.conv1_ru = nn.GELU()
        self.conv2_o = Dynamic_conv2d(256, num_classes, 3, padding=1)

    def forward(self, d4, d3, d2, d1):
        d4 = self.stage1(d4)
        d4 = self.stage1_ru(d4)
        d4 = self.up1(d4)
        d3 = self.stage2(d3)
        d3 = self.stage1_ru(d3)
        d3 = self.up2(d3)
        d2 = self.up3(d2)

        d_cat = torch.cat([d1, d2, d3, d4], dim=1)

        out = self.conv1_o(d_cat)
        out = self.conv1_ru(out)
        out = self.conv2_o(out)

        # out = self.up4(out)

        return F.sigmoid(out)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.GELU()

        self.deconv2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                     nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1),
                                     nn.BatchNorm2d(in_channels // 4))
        # self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        # self.norm2 = nn.BatchNorm2d(in_channels // 4)
        # self.relu2 = nonlinearity
        self.relu2 = nn.GELU()
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        # self.relu3 = nonlinearity
        self.relu3 = nn.GELU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        # x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DFITNet(nn.Module):
    def __init__(self, num_classes=9):
        super(DFITNet, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = ResNet50()
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.DFIMBlock1 = DFIMBlock(channels=256)
        self.DFIMBlock2 = DFIMBlock(channels=512)
        self.DFIMBlock3 = DFIMBlock(channels=1024)
        self.DFIM1 = nn.ModuleList([self.DFIMBlock1 for i in range(6)])
        self.DFIM2 = nn.ModuleList([self.DFIMBlock2 for i in range(4)])
        self.DFIM3 = nn.ModuleList([self.DFIMBlock3 for i in range(2)])

        # self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        # self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        # self.finalrelu1 = nonlinearity
        # self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalrelu2 = nonlinearity
        # self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.finaldeconv1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                        nn.Conv2d(256, 32, 3, padding=1))
        # self.finalconv1.apply(weights_init_kaiming)
        # self.finalrelu1 = nonlinearity
        self.finalrelu1 = nn.GELU()
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        # self.finalconv2.apply(weights_init_kaiming)
        # self.finalrelu2 = nonlinearity
        self.finalrelu2 = nn.GELU()
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        self.get_res = _Get_res()

    def forward(self, x, isres=False):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_R_MHSA_4 = self.encoder4(e3)

        # Center
        # e4 = self.dblock(e4)
        for i in range(2):
            e3_depthwise = self.DFIM3[i](e3)
        for i in range(4):
            e2_depthwise = self.DFIM2[i](e2)
        for i in range(6):
            e1_depthwise = self.DFIM1[i](e1)

        # Decoder
        d4 = self.decoder4(feature_R_MHSA_4) + e3_depthwise
        d3 = self.decoder3(d4) + e2_depthwise
        d2 = self.decoder2(d3) + e1_depthwise
        d1 = self.decoder1(d2)

        "Final Layers"

        out1 = d1

        out1 = self.finaldeconv1(out1)
        out1 = self.finalrelu1(out1)
        out1 = self.finalconv2(out1)
        out1 = self.finalrelu2(out1)
        out1 = self.finalconv3(out1)

        # return out1

        res = self.get_res(d4, d3, d2, d1)
        res4 = F.avg_pool2d(res, 8)
        res3 = F.avg_pool2d(res, 4)
        res2 = F.avg_pool2d(res, 2)
        res1 = F.avg_pool2d(res, 2)

        # step2
        # decoder2
        dd4 = self.decoder4(feature_R_MHSA_4) + e3_depthwise
        dd3 = self.decoder3(dd4 * (1 + res4)) + e2_depthwise
        dd2 = self.decoder2(dd3 * (1 + res3)) + e1_depthwise
        dd1 = self.decoder1(dd2 * (1 + res2))

        out = dd1 * (1 + res1) if isres else d1

        out = self.finaldeconv1(out)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out
