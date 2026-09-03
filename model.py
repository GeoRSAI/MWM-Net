import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
from ssim import msssim,ssim
import numpy as np
from wavelet import DWT_Haar, IWT_Haar
NUM_BANDS = 6
from cross import eca_layer,LDC,SS2D,DropPath
from enum import Enum, auto

# References:
#1.https://github.com/sunny2109/SAFMN
#2.https://github.com/millieXie/FusionMamba
#3.https://github.com/lixinghua5540/ECPW-STFN


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.ReplicationPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, stride=stride)
    )


# 主损失类（带加权）
class CompoundLoss(nn.Module):
    def __init__(self,
                 alpha=1.5,
                 lambda_wavelet=3,
                 normalize=True,model=None):
        super(CompoundLoss, self).__init__()
        self.alpha = alpha
        self.lambda_wavelet = lambda_wavelet
        self.normalize = normalize
        self.DWT = DWT_Haar()
        self.IWT = IWT_Haar()
        self.model = model

    def forward(self, prediction, target):
        # 小波损失
        prediction_LL, (prediction_HL, prediction_LH, prediction_HH) = self.DWT(prediction)
        target_LL, (target_HL, target_LH, target_HH) = self.DWT(target)

        LL_loss = F.l1_loss(prediction_LL, target_LL)
        HL_loss = F.l1_loss(prediction_HL, target_HL)
        LH_loss = F.l1_loss(prediction_LH, target_LH)
        HH_loss = F.l1_loss(prediction_HH, target_HH)
        wavelet_loss = LL_loss + HL_loss + LH_loss + HH_loss

        vision_loss = self.alpha * (1.0 - msssim(prediction, target,val_range=1.0, normalize=self.normalize))

        # 总损失
        loss = (
            self.lambda_wavelet * wavelet_loss +
            vision_loss
            +0.5 * F.l1_loss(prediction, target)
        )

        return loss

class Decoder(nn.Module):
    def __init__(self):
        channels = [105, 64, 48, NUM_BANDS]
        super(Decoder, self).__init__()
        self.layer1=conv3x3(channels[0], channels[1])
        self.relu1=nn.ReLU(True)
        self.layer2=conv3x3(channels[1], channels[2])
        self.relu2=nn.ReLU(True)
        self.layer3=nn.Conv2d(channels[2], channels[3],1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x=self.layer1(x)
        x=self.relu1(x)
        x=self.layer2(x)
        x=self.relu2(x)
        x=self.layer3(x)
        return self.sigmoid(x)

class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, 3,stride=stride)
        )

class CONV33(nn.Sequential):
   def __init__(self):
       channels = [NUM_BANDS, 24, 48, 64]
       super(CONV33, self).__init__(
           conv3x3(channels[0], channels[1]),
           nn.LeakyReLU(True),
           conv3x3(channels[1], channels[2]),
           nn.LeakyReLU(True),
           nn.Conv2d(channels[2], channels[3], 1),
           nn.LeakyReLU(True)
       )

class HFMamba(nn.Module):
    def __init__(self, channels=NUM_BANDS, d_state=16, expand=2, drop_path=0., step_size=2):
        super().__init__()
        self.channels = channels
        self.d_state = d_state
        self.expand = expand
        self.hidden_dim =int(channels * expand)

        # 核心组件定义
        self.norm = nn.LayerNorm(channels)
        self.eca = eca_layer(channels)
        self.ldc = LDC(in_channels=channels, out_channels=64)

        # 状态空间模块
        self.ssm = SS2D(
            d_model=channels,
            d_state=d_state,
            d_conv=3,
            expand=expand,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            step_size=step_size
        )
        self.conv_adjust = CONV33()
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)      #BHWC
        x=self.norm(x)
        x = x.permute(0, 3, 1, 2)     #BCHW

        # 路径1：LDC分支
        x1 = self.ldc(x)

        # 路径2：SSM+ECA分支
        x = x.permute(0, 2, 3, 1)
        x2 = self.norm(x).permute(0, 3, 1, 2)

        x2 = self.ssm(x2).permute(0, 3, 1, 2)

        x2 = x2.permute(0, 2, 3, 1)
        x2=  self.norm(x2).permute(0, 3, 1, 2)
        x2 = self.eca(x2)
        x2 = self.conv_adjust(x2)

        x3=x1+x2
        return x3

class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super(SAFM, self).__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels
        self.mfr = nn.ModuleList([nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim)for _ in range(n_levels)])
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)
        self.act = nn.GELU()
    def forward(self, x):
        h, w = x.size()[-2:]
        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            out.append(s)
        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        return out
class CCM(nn.Module):
    def __init__(self, dim):
        super(CCM, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim * 2, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(dim * 2, dim, 1, 1, 0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x
class FMM(nn.Module):
    def __init__(self, dim):
        super(FMM, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.safm = SAFM(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ccm = CCM(dim)
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.safm(x) + x
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        x = self.norm2(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.ccm(x) + x
        return x
class SAFMN(nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_blocks):
        super(SAFMN, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, dim, 3, 1, 1)
        self.fmm_blocks = nn.ModuleList([FMM(dim)for _ in range(num_blocks)])
        self.conv_out = nn.Conv2d(dim, out_channels, 3, 1, 1)
    def forward(self, x):
        x = self.conv_in(x)
        for block in self.fmm_blocks:
            x = block(x)
        x = self.conv_out(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ResidulBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidulBlock, self).__init__()
        residual = [
            Conv3X3WithPadding(in_channels, out_channels),
            nn.ReLU(True),
        ]
        ChannelA = [ChannelAttention(out_channels)]
        SpatialA = [SpatialAttention()]

        self.residual = nn.Sequential(*residual)
        self.ChannelA = nn.Sequential(*ChannelA)
        self.SpatialA = nn.Sequential(*SpatialA)

    def forward(self, inputs):
        residualfeature = self.residual(inputs)
        CA = self.ChannelA(residualfeature)
        channelrefined = residualfeature*CA
        SA = self.SpatialA(channelrefined)
        refined = channelrefined*SA
        return refined


#空间域
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)

class SpatialEncoder(nn.Module):
    def __init__(self, in_ch, base_ch=64):
        super().__init__()
        # Level 1: 原图大小 (H, W) -> 用于最终 Wavelet_result1 后的交互
        self.enc1 = ResBlock(in_ch, base_ch)

        # Level 2: (H/2, W/2) -> 用于 Wavelet_result2 后的交互
        self.enc2 = Down(base_ch, base_ch * 2)

        # Level 3: (H/4, W/4) -> 用于 Wavelet_result3 后的交互
        self.enc3 = Down(base_ch * 2, base_ch * 4)

    def forward(self, x):
        e1 = self.enc1(x)  # H, W
        e2 = self.enc2(e1)  # H/2, W/2
        e3 = self.enc3(e2)  # H/4, W/4
        return e1, e2, e3

#空频
class SFFusion(nn.Module):
    def __init__(self, spatial_ch, wavelet_ch):
        super().__init__()
        self.align = nn.Conv2d(spatial_ch, wavelet_ch, kernel_size=1) if spatial_ch != wavelet_ch else nn.Identity()

        self.norm = nn.InstanceNorm2d(wavelet_ch * 2)
        self.attn = nn.Sequential(
            nn.Conv2d(wavelet_ch * 2, wavelet_ch, kernel_size=3, padding=1),
            nn.InstanceNorm2d(wavelet_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(wavelet_ch, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Conv2d(wavelet_ch, wavelet_ch, kernel_size=3, padding=1)

    def forward(self, spatial_f, wavelet_f):
        spatial_f = self.align(spatial_f)
        concat_f = torch.cat([spatial_f, wavelet_f], dim=1)
        normalized_f = self.norm(concat_f)

        gate = self.attn(normalized_f)

        out = gate * spatial_f + (1 - gate) * wavelet_f

        return self.fuse_conv(out) + wavelet_f


class FusionNet_MWM(nn.Module):
    def __init__(self):
        super(FusionNet_MWM, self).__init__()
        self.decoder = Decoder()
        self.device = torch.device('cuda')
        self.DWT = DWT_Haar()
        self.IWT = IWT_Haar()
        self.HFMamba = HFMamba()
        self.LFEM = SAFMN(in_channels=NUM_BANDS, out_channels=64, dim=36, num_blocks=4)
        self.CBAM3=ResidulBlock(in_channels=128, out_channels=128)
        self.CBAM2 = ResidulBlock(in_channels=208, out_channels=208)
        self.CBAM1 = ResidulBlock(in_channels=228, out_channels=228)
        self.spatilblock=SpatialEncoder(in_ch=NUM_BANDS)

        self.fusion3 = SFFusion(256, 80)
        self.fusion2 = SFFusion(128, 100)
        self.fusion1 = SFFusion(64, 105)


    def forward(self, inputs):
        Land_ref = inputs[1]
        Modis_pre = inputs[-1]
        Modis_ref= inputs[0]

        res_data = Modis_pre - Land_ref

#空间域
        e1, e2, e3 = self.spatilblock(res_data)

#频率域
    #一级小波变换
        Land_L_1, (Land_H1_1, Land_H2_1, Land_H3_1) = self.DWT(Land_ref)
        Mo_L_1, (Mo_H1_1, Mo_H2_1, Mo_H3_1) = self.DWT(res_data)

    # 二级小波变换
        Land_L_2, (Land_H1_2, Land_H2_2, Land_H3_2) = self.DWT(Land_L_1)
        Mo_L_2, (Mo_H1_2, Mo_H2_2, Mo_H3_2) = self.DWT(Mo_L_1)

    # 三级小波变换
        Land_L_3, (Land_H1_3, Land_H2_3, Land_H3_3) = self.DWT(Land_L_2)
        Mo_L_3, (Mo_H1_3, Mo_H2_3, Mo_H3_3) = self.DWT(Mo_L_2)


    # Mo_L_3与Land_L_3、Land_H1_3, Land_H2_3, Land_H3_3特征提取
        Fea_Mo_L_3=self.LFEM(Mo_L_3)
        Fea_Land_L_3 = self.LFEM(Land_L_3)
        Fea_Land_H1_3 = self.HFMamba(Land_H1_3)
        Fea_Land_H2_3 = self.HFMamba(Land_H2_3)
        Fea_Land_H3_3 = self.HFMamba(Land_H3_3)

        # Mo_L_2与Land_L_2、Land_H1_2, Land_H2_2, Land_H3_2特征提取
        Fea_Mo_L_2 = self.LFEM(Mo_L_2)
        Fea_Land_L_2 = self.LFEM(Land_L_2)
        Fea_Land_H1_2 = self.HFMamba(Land_H1_2)
        Fea_Land_H2_2 = self.HFMamba(Land_H2_2)
        Fea_Land_H3_2 = self.HFMamba(Land_H3_2)

        # Mo_L_1与Fea_Land_L_1、Land_H1_1, Land_H2_1, Land_H3_1特征提取
        Fea_Mo_L_1 = self.LFEM(Mo_L_1)
        Fea_Land_L_1 = self.LFEM(Land_L_1)
        Fea_Land_H1_1 = self.HFMamba(Land_H1_1)
        Fea_Land_H2_1 = self.HFMamba(Land_H2_1)
        Fea_Land_H3_1 = self.HFMamba(Land_H3_1)

        L_3=torch.cat([Fea_Mo_L_3,Fea_Land_L_3],dim=1)

        L_3=self.CBAM3(L_3)


    #第一级逆变换
        Wavelet_result1 = self.IWT(torch.cat([L_3, Fea_Land_H1_3, Fea_Land_H2_3, Fea_Land_H3_3], dim=1))
        Wavelet_result1 = self.fusion3(e3, Wavelet_result1)

        L_2=torch.cat([Fea_Mo_L_2,Fea_Land_L_2,Wavelet_result1],dim=1)
        L_2=self.CBAM2(L_2)


    # 第二级逆变换
        Wavelet_result2 = self.IWT(torch.cat([L_2, Fea_Land_H1_2, Fea_Land_H2_2, Fea_Land_H3_2], dim=1))
        Wavelet_result2 = self.fusion2(e2, Wavelet_result2)

        L_1=torch.cat([Fea_Land_L_1,Fea_Mo_L_1,Wavelet_result2],dim=1)
        L_1 = self.CBAM1(L_1)

    # 第三级逆变换
        Wavelet_result3 = self.IWT(torch.cat([L_1, Fea_Land_H1_1, Fea_Land_H2_1, Fea_Land_H3_1], dim=1))

        Wavelet_result3 = self.fusion1(e1, Wavelet_result3)

        result = self.decoder(Wavelet_result3)

        return result



