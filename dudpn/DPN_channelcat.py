import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from efficientnet.utils import MemoryEfficientSwish, Swish
from efficientnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
# from nets.layers import Conv2dStaticSamePadding,Conv2dDynamicSamePadding,MaxPool2dStaticSamePadding
# from nets.layers import MemoryEfficientSwish, Swish

'''
    ①连接 ,feature  Map 分别经过dia_bn_relu 

    '''
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class Dilate_odd_block(nn.Module):
    def __init__(self,num_channels,hid_channels):
        super(Dilate_odd_block,self).__init__()
        self.conv_r3 = nn.Sequential(
            nn.Conv2d(num_channels, hid_channels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(hid_channels, momentum=0.01, eps=1e-3),
        )
        self.conv_r5 = nn.Sequential(
            nn.Conv2d(num_channels, hid_channels, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(hid_channels, momentum=0.01, eps=1e-3),
        )
        self.conv_r7 = nn.Sequential(
            nn.Conv2d(num_channels,hid_channels, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(hid_channels, momentum=0.01, eps=1e-3),
        )

    def forward(self,x):
        feat_r3 = self.conv_r3(x)
        feat_r3 = F.relu(feat_r3)
        feat_r5 = self.conv_r5(x)
        feat_r5 = F.relu(feat_r5)
        feat_r7 = self.conv_r7(x)
        feat_r7 = F.relu(feat_r7)
        return torch.cat((feat_r3, feat_r5, feat_r7), dim=1)


class Dilate_even_block(nn.Module):
    def __init__(self,num_channels,hid_channels):
        super(Dilate_even_block,self).__init__()
        self.conv_r2 = nn.Sequential(
            nn.Conv2d(num_channels,hid_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(hid_channels,momentum=0.01, eps=1e-3),
                                     )
        self.conv_r4 = nn.Sequential(
            nn.Conv2d(num_channels,hid_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(hid_channels, momentum=0.01, eps=1e-3),
        )
        self.conv_r6 = nn.Sequential(
            nn.Conv2d(num_channels,hid_channels, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(hid_channels, momentum=0.01, eps=1e-3),
        )

    def forward(self,x):
        feat_r2 = self.conv_r2(x)
        feat_r2 = F.relu(feat_r2)
        feat_r4 = self.conv_r4(x)
        feat_r4 = F.relu(feat_r4)
        feat_r6 = self.conv_r6(x)
        feat_r6 = F.relu(feat_r6)
        return torch.cat((feat_r2, feat_r4, feat_r6), dim=1)


class Dilate_merge(nn.Module):
    def __init__(self,num_channels,hid_channels):
        super(Dilate_merge,self).__init__()
        self.conv_r3 = nn.Sequential(
            nn.Conv2d(num_channels, hid_channels, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(hid_channels, momentum=0.01, eps=1e-3)
                                     )
        self.conv_r5 = nn.Sequential(
            nn.Conv2d(num_channels, hid_channels, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(hid_channels, momentum=0.01, eps=1e-3)
                                     )
        self.conv_r2 = nn.Sequential(
            nn.Conv2d(num_channels, hid_channels, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(hid_channels, momentum=0.01, eps=1e-3),
        )
        self.conv_r4 = nn.Sequential(
            nn.Conv2d(num_channels, hid_channels, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(hid_channels, momentum=0.01, eps=1e-3),
        )
        self.conv_r6 = nn.Sequential(
            nn.Conv2d(num_channels, hid_channels, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(hid_channels, momentum=0.01, eps=1e-3),
        )
    def forward(self,x):
        feat_r3 = self.conv_r3(x)
        feat_r3 = F.relu(feat_r3)
        feat_r5 = self.conv_r5(x)
        feat_r5 = F.relu(feat_r5)
        feat_r7 = self.conv_r7(x)
        feat_r7 = F.relu(feat_r7)
        feat_r2 = self.conv_r2(x)
        feat_r2 = F.relu(feat_r2)
        feat_r4 = self.conv_r4(x)
        feat_r4 = F.relu(feat_r4)
        feat_r6 = self.conv_r6(x)
        feat_r6 = F.relu(feat_r6)
        return torch.cat((feat_r2, feat_r3,feat_r4, feat_r5, feat_r6, feat_r7),dim=1)


class Dialate_even_down(nn.Module):
    def __init__(self,in_channels,hid_channels):
        super(Dialate_even_down,self).__init__()
        # mid_channels = in_channels//2
        # self.conv1x1 = nn.Sequential(
        #     nn.Conv2d(in_channels,mid_channels,kernel_size=1,stride=1),
        #     # nn.BatchNorm2d(mid_channels, momentum=0.01, eps=1e-3),
        #     nn.ReLU(),
        #     nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1),
        #     # nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3),
        #     nn.ReLU(),
        # )
        self.feat_even_block = Dilate_even_block(in_channels, hid_channels)
        # self.bn = nn.BatchNorm2d(3*hid_channels, momentum=0.01, eps=1e-3)
        self.down_channel =nn.Sequential(
            Conv2dStaticSamePadding(3*hid_channels,hid_channels, 1),
            nn.BatchNorm2d(hid_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(),
                                         )
    def forward(self,x):
        x=F.relu(x)
        # x=self.conv1x1(x)
        x=self.feat_even_block(x)
        # x=self.bn(x)
        # x=F.relu(x)
        out=self.down_channel(x)
        return out


class D_FPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
    # def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False,attention=True,use_p8=False):
        super(D_FPN, self).__init__()
        self.epsilon = epsilon

        # mid_channels = num_channels // 2
        # self.use_p8 = use_p8
        # even-diate
        self.feat1_even_block = Dialate_even_down(num_channels, num_channels)
        self.feat2_even_block = Dialate_even_down(num_channels, num_channels)
        # self.feat3_even_block = Dialate_even_down(num_channels, num_channels)
        # self.feat4_even_block =Dialate_even_down(num_channels, num_channels)
        # self.feat5_even_block = Dialate_even_down(num_channels, num_channels)

        # odd-diate
        # self.feat1_odd_block = Dilate_odd_block(num_channels, num_channels)
        # self.feat2_odd_block = Dilate_odd_block(num_channels, num_channels)
        # self.feat3_odd_block = Dilate_odd_block(num_channels, num_channels)
        # self.feat4_odd_block = Dilate_odd_block(num_channels, num_channels)
        # self.feat5_odd_block = Dilate_odd_block(num_channels, num_channels)

        # merge-diate
        # self.feat1_merge = Dilate_merge(num_channels,num_channels)
        # self.feat2_merge = Dilate_merge(num_channels,num_channels)
        # self.feat3_merge = Dilate_merge(num_channels,num_channels)

        # down-sample:得到F4,F5
        # self.f3_to_f4 = nn.Sequential(
        #     Conv2dStaticSamePadding(num_channels, num_channels, kernel_size=1),
        #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
        #     MaxPool2dStaticSamePadding(3, 2)
        # )
        #
        # self.f4_to_f5 = nn.Sequential(
        #     MaxPool2dStaticSamePadding(3, 2)
        # )

    # _____________________________________________________________________________
        self.first_time = first_time
        if self.first_time:
            # 获取到了efficientnet的最后三层，对其进行通道的下压缩
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p2_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            self.p1_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )

            # 对输入进来的p5进行宽高的下采样
            self.f3_to_f4 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.f4_to_f5 = nn.Sequential(
                MaxPool2dStaticSamePadding(3, 2)
            )

            # BIFPN第一轮的时候，跳线那里并不是同一个in
            self.p2_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
            # self.p4_down_channel = nn.Sequential(
            #     Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
            #     nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            # )
            self.p3_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
                nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            )
    # _____________________________________________________________________________
        # 第一大块儿
        # Up_sample

        self.p5_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_up = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_up = nn.Upsample(scale_factor=2, mode='nearest')

        self.p5_up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_up2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv1_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv2_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.p1d_down = MaxPool2dStaticSamePadding(3, 2)
        self.p2d_down = MaxPool2dStaticSamePadding(3, 2)
        self.p3d_down = MaxPool2dStaticSamePadding(3, 2)
        self.p4d_down = MaxPool2dStaticSamePadding(3, 2)

        self.conv1_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv2_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv3_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
        self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)

        self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

        # 简易注意力机制的weights
        # self.p5_up_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # self.p5_up_w1_relu = nn.ReLU()
        self.p4_up_w1 = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.p4_up_w1_relu = nn.ReLU()
        self.p3_up_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_up_w1_relu = nn.ReLU()
        self.p2_up_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_up_w1_relu = nn.ReLU()
        self.p1_up_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_up_w1_relu = nn.ReLU()

    #  ① 横向连接
        self.p5_down_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_down_w2_relu = nn.ReLU()
        self.p4_down_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_down_w2_relu = nn.ReLU()
        self.p3_down_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_down_w2_relu = nn.ReLU()
        self.p2_down_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_down_w2_relu = nn.ReLU()
        self.p2_down_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_down_w2_relu = nn.ReLU()
        self.p1_down_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_down_w2_relu = nn.ReLU()

    #     self.f5_up_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    #     self.f5_up_w1_relu = nn.ReLU()
    #     self.f4_up_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    #     self.f4_up_w1_relu = nn.ReLU()
    #     self.f3_up_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    #     self.f3_up_w1_relu = nn.ReLU()
    #     self.f2_up_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    #     self.f2_up_w1_relu = nn.ReLU()
    #     self.p1_up_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    #     self.p1_up_w1_relu = nn.ReLU()
    #
    #   # 横向skip连接
    #     self.p5_td_w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
    #     self.p5_td_w1_relu = nn.ReLU()
    #     self.p4_td_w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
    #     self.p4_td_w1_relu = nn.ReLU()
    #     self.p3_td_w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
    #     self.p3_td_w1_relu = nn.ReLU()
    #     self.p2_td_w1 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
    #     self.p2_td_w1_relu = nn.ReLU()
    # # 下采样连接
    #     self.f5_down_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    #     self.f5_down_w2_relu = nn.ReLU()
    #     self.f4_down_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    #     self.f4_down_w2_relu = nn.ReLU()
    #     self.f3_down_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    #     self.f3_down_w2_relu = nn.ReLU()
    #     self.f2_down_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
    #     self.f2_down_w2_relu = nn.ReLU()

    # Pn_f -> Pn_d 横向连接
    #
    #     self.pf1_conv = nn.Conv2d(num_channels,num_channels,kernel_size=1)
    #     self.pf2_conv = nn.Conv2d(num_channels,num_channels,kernel_size=1)
    #     self.pf3_conv = nn.Conv2d(num_channels,num_channels,kernel_size=1)
    #     self.pf4_conv = nn.Conv2d(num_channels,num_channels,kernel_size=1)

    # 第二大块
    # dia_conv_even
    #     self.p1f_dia = Dialate_even_down(num_channels, num_channels)
    #     self.p2f_dia = Dialate_even_down(num_channels, num_channels)
    #     self.p3f_dia = Dialate_even_down(num_channels, num_channels)
    #     self.p4f_dia = Dialate_even_down(num_channels, num_channels)
    # # dia_conv_odd
    #     self.p1f_dia = Dilate_odd_block(int(num_channels/2), int(num_channels/2))
    #     self.p2f_dia = Dilate_odd_block(int(num_channels/2), int(num_channels/2))
    #     self.p3f_dia = Dilate_odd_block(int(num_channels/2), int(num_channels/2))
    #     self.p4f_dia = Dilate_odd_block(int(num_channels/2), int(num_channels/2))

        # self.p1d_dia = Dialate_even_down(num_channels, num_channels)
        # self.p2d_dia = Dialate_even_down(num_channels, num_channels)
        # self.p3d_dia = Dialate_even_down(num_channels, num_channels)
        # self.p4d_dia = Dialate_even_down(num_channels, num_channels)
        # self.p5d_dia = Dialate_even_down(num_channels, num_channels)

        self.attention = attention

    def forward(self, inputs):
        """ Dfpn模块结构示意图
                    --------->f1_rate3----
                   |                      |
                   |                      |
            F1--------------->f1_rate5---------> f1_cat---->depth_conv---->p1_odd(p1_even、p1_merge)
                   |                     |
                   |                     |
                   ---------->f1_rate7 ---


            P1----------->|         P1_f----------------------------------------------P1_f--------------P1_d
                                     |                                                                   |
                          |-----> Cat  ←←--------                                                        |
                          |                     |                                                        |
                                                |                                                        |
                       p2_up                    |
                          |                    p3_up2                                                p1_down_sample
                     up_sample                   |                                                      |
                          |                   up_sample                                                 |
            P2----------  |                      |-------------------------------------P2_f------------Cat
                                      ------→→Cat←←------                                              P2_conv_d
                                      |                 |                                               |
                                      |                 |                                               |
                                    P3_up             P4_up2                                        p2_down_sample
                                      |                 ↑                                               |
                                up_sample            Up_sample                                          |
                                      |                |-------------------------------P3_f----------- Cat
            P3------------------------           -----→→Cat←←------                                    P3_conv_d
                                                 |   (attention）    |                                  |
                                            P4_up                 P5_up2                               |
                                                 |                  |                                p3_down_sample
                                                 |                  |                                  |
                                            Up_sample           upsample                               |
                                                 ↑                 ↑                                   |
                                                 |                 ↑
            P4-----------------------------------|               P5_up---------------------------------Cat
                                                                   ↑                                   P4_conv_d
                                                                   |                                   |
                                                                   |                                   |
                                                                UP_sample                            p4_down_sample
                                                                   |                                   |
            P5-----------------------------------------------------                                   P5_d
        """
        # if self.attention:
        #     p1_out, p2_out, p3_out, p4_out, p5_out = self._forward_fast_attention(inputs)
        # else:
        #     p1_out, p2_out, p3_out, p4_out, p5_out = self._forward(inputs)
        #
        # return p1_out, p2_out, p3_out, p4_out, p5_out
        if self.attention:
            outs = self._forward_fast_attention(inputs)
        else:
            outs = self._forward(inputs)

        return outs

    def _forward_fast_attention(self, inputs):
        # 当phi=1、2、3、4、5的时候使用_forward_fast_attention
        if self.first_time:
            # 第一次BIFPN需要下采样与降通道获得
            # p3_in p4_in p5_in p6_in p7_in
            f1, f2, f3 = inputs
            # p1_in = self.p3_down_channel(p3)

            # p4_in_1 = self.p4_down_channel(p4)
            # p4_in_2 = self.p4_down_channel_2(p4)
            #
            # p5_in_1 = self.p5_down_channel(p5)
            # p5_in_2 = self.p5_down_channel_2(p5)
            p1 = self.p1_down_channel(f1)
            p2 = self.p2_down_channel(f2)
            p3 = self.p3_down_channel(f3)

            # p1 = self.feat1_even_block(f1_in)
            # p2 = self.feat2_even_block(f2_in)
            # p3 = self.feat3_even_block(f3_in)

            p4 = self.f3_to_f4(f3)    # p4_1 未进行 膨胀卷积
            p5 = self.f4_to_f5(p4)  # p5_1 未进行 膨胀卷积

            # p4 = self.feat4_even_block(p4_1)
            # p5 = self.feat5_even_block(p5_1)

            # 简单的注意力机制，用于确定更关注p7_in还是p6_in
            p4_up_w1 = self.p4_up_w1_relu(self.p4_up_w1)
            weight = p4_up_w1 / (torch.sum(p4_up_w1, dim=0) + self.epsilon)
            p4f = self.conv4_up(self.swish(weight[0] * self.p5_up(p5)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p3_up_w1 = self.p3_up_w1_relu(self.p3_up_w1)
            weight = p3_up_w1 / (torch.sum(p3_up_w1, dim=0) + self.epsilon)
            p3f = self.conv3_up(self.swish(weight[0] * self.p4_up(p4) + weight[1] * self.p5_up2(p4f)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p2_up_w1 = self.p2_up_w1_relu(self.p2_up_w1)
            weight = p2_up_w1 / (torch.sum(p2_up_w1, dim=0) + self.epsilon)
            p2f = self.conv2_up(self.swish(weight[0] * self.p3_up(p3) + weight[1] * self.p4_up2(p3f)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p1_up_w1 = self.p1_up_w1_relu(self.p1_up_w1)
            weight = p1_up_w1 / (torch.sum(p1_up_w1, dim=0) + self.epsilon)
            p1f = self.conv1_up(self.swish(weight[0] * self.p2_up(p2) + weight[1] * self.p3_up2(p2f)))

        #  第一次横向连接_____

            # p1d_in  = self.pf1_conv(p1f)
            # p2d_in1 = self.pf2_conv(p2f)
            # p3d_in1 = self.pf3_conv(p3f)
            # p4d_in1 = self.pf4_conv(p4f)

        #  dialte_conv--
            p1d_in_dia = self.feat1_even_block(p1f)
            p2d_in_dia = self.feat2_even_block(p2f)
            p3d_in_dia = p3f
            p4d_in_dia = p4f



            # Pn_d下采样

            # self.p1_down_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            # self.p1_down_w2_relu = nn.ReLU()
            p1_down_w2 = self.p1_down_w2_relu(self.p1_down_w2)
            weight = p1_down_w2/(torch.sum(p1_down_w2, dim=0) + self.epsilon)
            p1d = self.conv1_down(self.swish(weight[0]*p1d_in_dia+weight[1]*p1))

            p2_down_w2 =self.p2_down_w2_relu(self.p2_down_w2)
            weight = p2_down_w2/(torch.sum(p2_down_w2,dim=0)+self.epsilon)
            p2d = self.conv2_down(self.swish(weight[0]*p2d_in_dia+weight[1]*p2+weight[2]*self.p1d_down(p1d)))

            p3_down_w2 =self.p3_down_w2_relu(self.p3_down_w2)
            weight = p3_down_w2/(torch.sum(p3_down_w2,dim=0)+self.epsilon)
            p3d = self.conv3_down(self.swish(weight[0]*p3d_in_dia+weight[1]*p3+weight[2]*self.p2d_down(p2d)))

            p4_down_w2 =self.p4_down_w2_relu(self.p4_down_w2)
            weight = p4_down_w2/(torch.sum(p4_down_w2,dim=0)+self.epsilon)
            p4d = self.conv4_down(self.swish(weight[0]*p4d_in_dia+weight[1]*p4+weight[2]*self.p3d_down(p3d)))

            p5_down_w2 = self.p5_down_w2_relu(self.p5_down_w2)
            weight = p5_down_w2/(torch.sum(p5_down_w2, dim=0) + self.epsilon)
            p5d = self.conv5_down(self.swish(weight[0]*p5+weight[1]*self.p4d_down(p4d)))


            p1_out = p1d
            p2_out = p2d
            p3_out = p3d
            p4_out = p4d
            p5_out = p5d


        else:
            p1, p2, p3, p4, p5 = inputs

            p4_up_w1 = self.p4_up_w1_relu(self.p4_up_w1)
            weight = p4_up_w1 / (torch.sum(p4_up_w1, dim=0) + self.epsilon)
            p4f = self.conv4_up(self.swish(weight[0] * self.p5_up(p5)))

            # 简单的注意力机制，用于确定更关注p6_up还是p5_in
            p3_up_w1 = self.p3_up_w1_relu(self.p3_up_w1)
            weight = p3_up_w1 / (torch.sum(p3_up_w1, dim=0) + self.epsilon)
            p3f = self.conv3_up(self.swish(weight[0] * self.p4_up(p4) + weight[1] * self.p5_up2(p4f)))

            # 简单的注意力机制，用于确定更关注p5_up还是p4_in
            p2_up_w1 = self.p2_up_w1_relu(self.p2_up_w1)
            weight = p2_up_w1 / (torch.sum(p2_up_w1, dim=0) + self.epsilon)
            p2f = self.conv2_up(self.swish(weight[0] * self.p3_up(p3) + weight[1] * self.p4_up2(p3f)))

            # 简单的注意力机制，用于确定更关注p4_up还是p3_in
            p1_up_w1 = self.p1_up_w1_relu(self.p1_up_w1)
            weight = p1_up_w1 / (torch.sum(p1_up_w1, dim=0) + self.epsilon)
            p1f = self.conv1_up(self.swish(weight[0] * self.p2_up(p2) + weight[1] * self.p3_up2(p2f)))

            #  第一次横向连接_____

            # p1d_in = self.pf1_conv(p1f)
            # p2d_in1 = self.pf2_conv(p2f)
            # p3d_in1 = self.pf3_conv(p3f)
            # p4d_in1 = self.pf4_conv(p4f)

            #  dialte_conv--
            # p1d_in_dia = self.feat1_even_block(p1f)
            p1d_in_dia = self.feat1_even_block(p1f)
            p2d_in_dia = self.feat2_even_block(p2f)
            p3d_in_dia = p3f
            p4d_in_dia = p4f



            # Pn_d下采样

            # self.p1_down_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            # self.p1_down_w2_relu = nn.ReLU()
            p1_down_w2 = self.p1_down_w2_relu(self.p1_down_w2)
            weight = p1_down_w2 / (torch.sum(p1_down_w2, dim=0) + self.epsilon)
            p1d = self.conv1_down(self.swish(weight[0] * p1d_in_dia + weight[1] * p1))

            p2_down_w2 = self.p2_down_w2_relu(self.p2_down_w2)
            weight = p2_down_w2 / (torch.sum(p2_down_w2, dim=0) + self.epsilon)
            p2d = self.conv2_down(self.swish(weight[0] * p2d_in_dia + weight[1] * p2 + weight[2] * self.p1d_down(p1d)))

            p3_down_w2 = self.p3_down_w2_relu(self.p3_down_w2)
            weight = p3_down_w2 / (torch.sum(p3_down_w2, dim=0) + self.epsilon)
            p3d = self.conv3_down(self.swish(weight[0] * p3d_in_dia + weight[1] * p3 + weight[2] * self.p2d_down(p2d)))

            p4_down_w2 = self.p4_down_w2_relu(self.p4_down_w2)
            weight = p4_down_w2 / (torch.sum(p4_down_w2, dim=0) + self.epsilon)
            p4d = self.conv4_down(self.swish(weight[0] * p4d_in_dia + weight[1] * p4 + weight[2] * self.p3d_down(p3d)))

            p5_down_w2 = self.p5_down_w2_relu(self.p5_down_w2)
            weight = p5_down_w2 / (torch.sum(p5_down_w2, dim=0) + self.epsilon)
            p5d = self.conv5_down(self.swish(weight[0] * p5 + weight[1] * self.p4d_down(p4d)))


            p1_out = p1d
            p2_out = p2d
            p3_out = p3d
            p4_out = p4d
            p5_out = p5d

        return p1_out, p2_out, p3_out, p4_out, p5_out

    def _forward(self, inputs):
        # 当phi=6、7的时候使用_forward
        if self.first_time:
            '''
            # 第一次BIFPN需要下采样与降通道获得
            # p3_in p4_in p5_in p6_in p7_in
            # p3, p4, p5 = inputs
            # p3_in = self.p3_down_channel(p3)
            # p4_in_1 = self.p4_down_channel(p4)
            # p4_in_2 = self.p4_down_channel_2(p4)
            # p5_in_1 = self.p5_down_channel(p5)
            # p5_in_2 = self.p5_down_channel_2(p5)
            # p6_in = self.p5_to_p6(p5)
            # p7_in = self.p6_to_p7(p6_in)
            '''

            f1, f2, f3 = inputs
            # p1_in = self.p3_down_channel(p3)

            # p4_in_1 = self.p4_down_channel(p4)
            # p4_in_2 = self.p4_down_channel_2(p4)
            #
            # p5_in_1 = self.p5_down_channel(p5)
            # p5_in_2 = self.p5_down_channel_2(p5)

            p1 = self.p1_down_channel(f1)
            p2 = self.p2_down_channel(f2)
            p3 = self.p3_down_channel(f3)

            # p1 = self.feat1_even_block(f1_in)
            # p2 = self.feat2_even_block(f2_in)
            # p3 = self.feat3_even_block(f3_in)

            p4 = self.f3_to_f4(f3)  # p4_1 未进行 膨胀卷积
            p5 = self.f4_to_f5(p4)  # p5_1 未进行 膨胀卷积

            # p4 = self.feat4_even_block(p4_1)
            # p5 = self.feat5_even_block(p5_1)

            p4f = self.conv4_up(self.swish(self.p5_up(p5)))

            p3f = self.conv3_up(self.swish(self.p4_up(p4) + self.p5_up2(p4f)))

            p2f = self.conv2_up(self.swish(self.p3_up(p3) + self.p4_up2(p3f)))

            p1f = self.conv1_up(self.swish(self.p2_up(p2) + self.p3_up2(p2f)))

            #  第一次横向连接_____

            # p1d_in = self.pf1_conv(p1f)
            # p2d_in1 = self.pf2_conv(p2f)
            # p3d_in1 = self.pf3_conv(p3f)
            # p4d_in1 = self.pf4_conv(p4f)

            #  dialte_conv--
            # p1d_in_dia = self.feat1_even_block(p1f)
            p1d_in_dia = self.feat1_even_block(p1f)
            p2d_in_dia = self.feat2_even_block(p2f)
            p3d_in_dia = p3f
            p4d_in_dia = p4f


            p1d = self.conv1_down(self.swish(p1d_in_dia + p1))

            p2d = self.conv2_down(self.swish(p2d_in_dia + p2 + self.p1d_down(p1d)))

            p3d = self.conv3_down(self.swish(p3d_in_dia + p3 + self.p2d_down(p2d)))

            p4d = self.conv4_down(self.swish(p4d_in_dia + p4 + self.p3d_down(p3d)))

            p5d = self.conv5_down(self.swish(p5 + self.p4d_down(p4d)))


            p1_out = p1d
            p2_out = p2d
            p3_out = p3d
            p4_out = p4d
            p5_out = p5d
        else:

            p1, p2, p3, p4, p5 = inputs

            p4f = self.conv4_up(self.swish(self.p5_up(p5)))

            p3f = self.conv3_up(self.swish(self.p4_up(p4) + self.p5_up2(p4f)))

            p2f = self.conv2_up(self.swish(self.p3_up(p3) + self.p4_up2(p3f)))

            p1f = self.conv1_up(self.swish(self.p2_up(p2) + self.p3_up2(p2f)))

            #  第一次横向连接_____

            # p1d_in = self.pf1_conv(p1f)
            # p2d_in1 = self.pf2_conv(p2f)
            # p3d_in1 = self.pf3_conv(p3f)
            # p4d_in1 = self.pf4_conv(p4f)

            #  dialte_conv--
            # p1d_in_dia = self.p1f_dia(p1f)
            # p2d_in_dia = self.p2f_dia(p2f)
            # p1d_in_dia = self.feat1_even_block(p1f)
            p1d_in_dia = self.feat1_even_block(p1f)
            p2d_in_dia = self.feat2_even_block(p2f)
            p3d_in_dia = p3f
            p4d_in_dia = p4f



            p1d = self.conv1_down(self.swish(p1d_in_dia + p1))

            p2d = self.conv2_down(self.swish(p2d_in_dia + p2 + self.p1d_down(p1d)))

            p3d = self.conv3_down(self.swish(p3d_in_dia + p3 + self.p2d_down(p2d)))

            p4d = self.conv4_down(self.swish(p4d_in_dia + p4 + self.p3d_down(p3d)))

            p5d = self.conv5_down(self.swish(p5 + self.p4d_down(p4d)))

            p1_out = p1d
            p2_out = p2d
            p3_out = p3d
            p4_out = p4d
            p5_out = p5d

        return p1_out, p2_out, p3_out, p4_out, p5_out

# net =D_FPN()