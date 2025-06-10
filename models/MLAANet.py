import torch.nn.functional as F
import torch
import torch.nn as nn
from .base import *
from .DWT_IDWT_layer import DWT_2D
from .ECAEModule import HSAttention, CSAttention, WSAttention
from torch.nn.parameter import Parameter
class MLAANet(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, img_ch=3, output_ch=1):
        super(MLAANet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        #
        # self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #

        self.Hypool1 = Hypool(in_channels=filters[0], out_channels=filters[0])
        self.Hypool2 = Hypool(in_channels=filters[1], out_channels=filters[1])
        self.Hypool3 = Hypool(in_channels=filters[2], out_channels=filters[2])
        self.Hypool4 = Hypool(in_channels=filters[3], out_channels=filters[3])

        self.Conv1 = DoubleConvBlock(img_ch, filters[0])
        self.Conv2 = DoubleConvBlock(filters[0], filters[1])
        self.Conv3 = DoubleConvBlock(filters[1], filters[2])
        self.Conv4 = DoubleConvBlock(filters[2], filters[3])
        self.Conv5 = DoubleConvBlock(filters[3], filters[4])

        self.MFRA = MFRA(filters[3], filters[4])

        self.CSA1 = CSA(in_planes=[filters[1], filters[2], filters[2]], kernel_size=5)
        self.CSA2 = CSA(in_planes=[filters[2], filters[1], filters[1]], kernel_size=5)
        self.CSA3 = CSA(in_planes=[filters[3], filters[0], filters[0]], kernel_size=5)
        self.CSA4 = CSA(in_planes=[filters[4], 32, 32], kernel_size=7)

        self.PDC_E1 = PDC_E(filters[3], dropout_rate=0.5)
        self.PDC_E2 = PDC_E(filters[2], dropout_rate=0.5)
        self.PDC_E3 = PDC_E(filters[1], dropout_rate=0.5)
        self.PDC_E4 = PDC_E(filters[0], dropout_rate=0.5)

        self.PDC_D1 = PDC_D(filters[4], dropout_rate=0.5)
        self.PDC_D2 = PDC_D(filters[3], dropout_rate=0.5)
        self.PDC_D3 = PDC_D(filters[2], dropout_rate=0.5)
        self.PDC_D4 = PDC_D(filters[1], dropout_rate=0.5)

        self.Up5 = UpConv(filters[4], filters[3])
        self.Up_conv5 = DoubleConvBlock(filters[4], filters[3])

        self.Up4 = UpConv(filters[4], filters[2])
        self.Up_conv4 = DoubleConvBlock(filters[3], filters[2])

        self.Up3 = UpConv(filters[3], filters[1])
        self.Up_conv3 = DoubleConvBlock(filters[2], filters[1])

        self.Up2 = UpConv(filters[2], filters[0])
        self.Up_conv2 = DoubleConvBlock(filters[1], filters[0])

        self.Conv = nn.Conv2d(
            filters[1], output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        e1 = self.Conv1(x)
        e1 = self.PDC_E4(e1)

        e2 = self.Hypool1(e1)
        e2 = self.Conv2(e2)
        e2 = self.PDC_E3(e2)

        e3 = self.Hypool2(e2)
        e3 = self.Conv3(e3)
        e3 = self.PDC_E2(e3)

        e4 = self.Hypool3(e3)
        e4 = self.Conv4(e4)
        e4 = self.PDC_E1(e4)

        e5 = self.Hypool4(e4)
        e5 = self.MFRA(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.CSA4(d5)
        d5 = self.PDC_D1(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.CSA3(d4)
        d4 = self.PDC_D2(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.CSA2(d3)
        d3 = self.PDC_D3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.CSA1(d2)
        d2 = self.PDC_D4(d2)

        out = self.Conv(d2)
        return out

    def name(self):
        return "MLAANet"

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.conv_block(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.double_conv_block = nn.Sequential(
            ConvBlock(self.in_channels, self.out_channels),
            # ConvBlock(self.out_channels, self.out_channels),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.double_conv_block(x)
        return x


class ConvBlock1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.conv_block(x)
        return x



class DoubleConvBlock1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConvBlock1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.double_conv_block = nn.Sequential(
            ConvBlock1(self.in_channels, self.out_channels),
            ConvBlock1(self.out_channels, self.out_channels),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.double_conv_block(x)
        return x

class CSA(nn.Module):
    def __init__(self, kernel_size=7, in_planes=None, ratio=8, mac_pattern=0, mic_pattern=0):
        super(CSA, self).__init__()
        self.cs = ShuffleAttention(channel=in_planes[0])
        self.hs = ShuffleAttentionh(channel=in_planes[1])
        self.ws = ShuffleAttentionw(channel=in_planes[2])
        self.bn = nn.BatchNorm2d(in_planes[0])
        self.relu = nn.ReLU()
        self.pattern = mac_pattern
        # self.mic_pattern = mic_pattern

    def forward(self, x):
        if self.pattern == 0: # All summed up
            x0 = self.cs(x)
            x1 = self.hs(x)
            x2 = self.ws(x)
            out = (x0 + x1 + x2)/3.0
        elif self.pattern == 1: # All in a row
            x0 = self.cs(x)
            x1 = self.hs(x0)
            out = self.ws(x1)
        elif self.pattern == 2:
            x0 = self.cs(x)
            x1 = self.hs(x0)
            x2 = self.ws(x0)
            out = 0.5*(x1 + x2)
        elif self.pattern == 3:
            x0 = self.cs(x)
            x1 = self.hs(x0)
            x2 = self.ws(x0)
            out = (x0 + x1 + x2)/3.0
        out = self.relu(out)
        out = self.bn(out)
        return out

class ShuffleAttention(nn.Module):

    def __init__(self, channel=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out

class ShuffleAttentionh(nn.Module):

    def __init__(self, channel=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b1, c1, h1, w1 = x.size()
        x = torch.reshape(x, (b1, h1, c1, w1))
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        out = torch.reshape(out, (b1, c1, h1, w1))
        return out
class ShuffleAttentionw(nn.Module):

    def __init__(self, channel=512,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b1, c1, h1, w1 = x.size()
        x = torch.reshape(x, (b1, w1, h1, c1))
        b, c, h, w = x.size()
        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w

        #channel_split
        x_0,x_1=x.chunk(2,dim=1) #bs*G,c//(2*G),h,w

        #channel attention
        x_channel=self.avg_pool(x_0) #bs*G,c//(2*G),1,1
        x_channel=self.cweight*x_channel+self.cbias #bs*G,c//(2*G),1,1
        x_channel=x_0*self.sigmoid(x_channel)

        #spatial attention
        x_spatial=self.gn(x_1) #bs*G,c//(2*G),h,w
        x_spatial=self.sweight*x_spatial+self.sbias #bs*G,c//(2*G),h,w
        x_spatial=x_1*self.sigmoid(x_spatial) #bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out=torch.cat([x_channel,x_spatial],dim=1)  #bs*G,c//G,h,w
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        out = torch.reshape(out, (b1, c1, h1, w1))
        return out

class Hypool(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Hypool, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.wave_pool = DWT_2D(wavename='haar')

    def forward(self, x):
        x_max = self.max_pool(x)
        x_wave = self.wave_pool(x)
        # x_wave = torch.tensor(x_wave)
        # print(x_wave.shape)
        x = self.conv2d(x_wave[0]) + x_max
        # x_concat = torch.cat((x, x_max), 1)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class ACM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ACM, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MFRA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFRA, self).__init__()
        self.ascpp1 = ACM(in_channels, out_channels, kernel_size=1, padding=0)
        self.ascpp2 = ACM(in_channels, out_channels, kernel_size=3, padding=1)
        self.ascpp3 = ACM(in_channels, out_channels, kernel_size=5, padding=2)
        self.ascpp4 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.CBAM = CBAM(out_channels*3, reduction_ratio=16)

    def forward(self, x):
        x1 = self.ascpp1(x)
        x2 = self.ascpp2(x)
        x3 = self.ascpp3(x)
        # x3 = self.CBAM(x3)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.CBAM(x)
        x = self.ascpp4(x)
        return x


class DepthWiseConv2D(nn.Module):
    def __init__(self, in_channels, dilation, kernel_size, padding, stride=(1, 1)):
        super(DepthWiseConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding,
                                   stride=stride, groups=in_channels, dilation=dilation)

    def forward(self, x):
        return self.depthwise(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + x
        out = F.relu(out)

        return out

class PDC_E(nn.Module):
    def __init__(self, in_channels, dropout_rate):
        super(PDC_E, self).__init__()
        self.conv1_1 = DepthWiseConv2D(in_channels, dilation=1, kernel_size=(1, 7), padding=(0, 3), stride=(1, 1))
        self.conv1_2 = DepthWiseConv2D(in_channels, dilation=1, kernel_size=(7, 1), padding=(3, 0), stride=(1, 1))
        self.conv2_1 = DepthWiseConv2D(in_channels, dilation=2, kernel_size=(1, 7), padding=(0, 6), stride=(1, 1))
        self.conv2_2 = DepthWiseConv2D(in_channels, dilation=2, kernel_size=(7, 1), padding=(6, 0), stride=(1, 1))
        self.conv3_1 = DepthWiseConv2D(in_channels, dilation=3, kernel_size=(1, 7), padding=(0, 9), stride=(1, 1))
        self.conv3_2 = DepthWiseConv2D(in_channels, dilation=3, kernel_size=(7, 1), padding=(9, 0), stride=(1, 1))
        self.dw = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.Res = ResidualBlock(in_channels)

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.conv1_2(out1)
        out2 = self.conv2_1(x)
        out2 = self.conv2_2(out2)
        out3 = self.conv3_1(x)
        out3 = self.conv3_2(out3)
        out = out1 + out2 + out3

        out = self.conv(out)
        out = self.relu(out)
        out = self.dropout(out)
        out0 = self.dw(x)
        out1 = self.Res(x)
        out = out0 + out1 + out

        return out

class PDC_D(nn.Module):
    def __init__(self, in_channels, dropout_rate):
        super(PDC_D, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.dw = nn.Conv2d(in_channels, in_channels, 1, 1, 0, groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.Res = ResidualBlock(in_channels)

    def forward(self, x):
        out1 = self.conv1_1(x)
        # out2 = self.conv2_1(x)
        out2 = self.conv2_2(out1)
        # out3 = self.conv3_1(x)
        out3 = self.conv3_2(out2)
        out = out1 + out2 + out3

        out = self.conv(out)
        out = self.relu(out)
        out = self.dropout(out)
        out0 = self.dw(x)
        out1 = self.Res(x)
        out = out0 + out1 + out
        return out

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    image = torch.rand((2, 3, 256, 256)).to(DEVICE)
    model = MLAANet().to(DEVICE)
    res = model(image)
    print(res.shape)
