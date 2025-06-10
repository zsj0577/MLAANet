import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from collections import OrderedDict
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8, gamma=2, b=1, pattern=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.in_planes = in_planes
        # self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
        #                         nn.ReLU(),
        #                         nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        # self.sigmoid = nn.Sigmoid()
        kernel_size = int(abs((math.log(self.in_planes, 2) + b) / gamma))
        kernel_size = np.max([kernel_size, 3])
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        # self.channel_shuffle = ChannelShuffle(groups=groups)
        self.con1 = nn.Conv1d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)
        self.act1 = nn.Sigmoid()
        self.pattern = pattern

    def forward(self, x):
        if self.pattern == 0:
            out1 = self.avg_pool(x) + self.max_pool(x)
        elif self.pattern == 1:
            out1 = self.avg_pool(x)
        elif self.pattern == 2:
            out1 = self.max_pool(x)
        else:
            output1 = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
            output1 = self.con1(output1).transpose(-1, -2).unsqueeze(-1)

            output2 = self.max_pool(x).squeeze(-1).transpose(-1, -2)
            output2 = self.con1(output2).transpose(-1, -2).unsqueeze(-1)
            out1 = output1 + output2

        if self.pattern != 3:
            out1 = out1.squeeze(-1).transpose(-1, -2)
            out1 = self.con1(out1).transpose(-1, -2).unsqueeze(-1)

        output = self.act1(out1)
        # output = self.act1(out1)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class HSAttention(nn.Module):
    def __init__(self, in_planes=None, kernel_size=7, ratio=8, pattern=0):
        super(HSAttention, self).__init__()
        self.ca = ChannelAttention(in_planes=in_planes, ratio=ratio, pattern=3)
        # self.eca = ECAAttention()
        self.sa = SpatialAttention(kernel_size)
        self.bn = nn.BatchNorm2d(in_planes)
        self.p = pattern
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, W, H = x.shape
        y = torch.reshape(x, (B, H, W, C))
        if self.p == 0:
            y = y * self.ca(y)
            y = y * self.sa(y)
        elif self.p == 1:
            y = y * self.sa(y)
            y = y * self.ca(y)
        else:
            y0 = y * self.ca(y)
            y1 = y * self.sa(y)
            y = 0.5*(y0 + y1)
        y = self.bn(y)
        out = self.relu(y)
        out = torch.reshape(out, (B, C, W, H))
        return out
class HSAttention1(nn.Module):
    def __init__(self, in_planes=None, kernel_size=7, ratio=8, pattern=0):
        super(HSAttention1, self).__init__()
        # self.ca = ChannelAttention(in_planes=in_planes, ratio=ratio, pattern=3)
        self.eca = ECAAttention()
        self.sa = SpatialAttention(kernel_size)
        self.bn = nn.BatchNorm2d(in_planes)
        self.p = pattern
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, W, H = x.shape
        y = torch.reshape(x, (B, H, W, C))
        if self.p == 0:
            y = y * self.eca(y)
            y = y * self.sa(y)
        elif self.p == 1:
            y = y * self.sa(y)
            y = y * self.ca(y)
        else:
            y0 = y * self.ca(y)
            y1 = y * self.sa(y)
            y = 0.5*(y0 + y1)
        y = self.bn(y)
        out = self.relu(y)
        out = torch.reshape(out, (B, C, W, H))
        return out
class CSAttention1(nn.Module):
    def __init__(self, in_planes=None, kernel_size=7, ratio=8, pattern=0):
        super(CSAttention1, self).__init__()
        # self.ca = ChannelAttention(in_planes=in_planes, ratio=ratio, pattern=3)
        self.eca = ECAAttention()
        self.sa = SpatialAttention(kernel_size)
        self.bn = nn.BatchNorm2d(in_planes)
        self.p = pattern
        self.relu = nn.ReLU()

    def forward(self, x):
        y = x
        if self.p == 0:
            y = y * self.eca(y)
            y = y * self.sa(y)
        elif self.p == 1:
            y = y * self.sa(y)
            y = y * self.ca(y)
        else:
            y0 = y * self.ca(y)
            y1 = y * self.sa(y)
            y = 0.5*(y0 + y1)
        y = self.bn(y)
        out = self.relu(y)
        return out
class CSAttention(nn.Module):
    def __init__(self, in_planes=None, kernel_size=7, ratio=8, pattern=0):
        super(CSAttention, self).__init__()
        self.ca = ChannelAttention(in_planes=in_planes, ratio=ratio, pattern=3)
        # self.eca = ECAAttention()
        self.sa = SpatialAttention(kernel_size)
        self.bn = nn.BatchNorm2d(in_planes)
        self.p = pattern
        self.relu = nn.ReLU()

    def forward(self, x):
        y = x
        if self.p == 0:
            y = y * self.ca(y)
            y = y * self.sa(y)
        elif self.p == 1:
            y = y * self.sa(y)
            y = y * self.ca(y)
        else:
            y0 = y * self.ca(y)
            y1 = y * self.sa(y)
            y = 0.5*(y0 + y1)
        y = self.bn(y)
        out = self.relu(y)
        return out
class WSAttention(nn.Module):
    def __init__(self, in_planes=None, kernel_size=7, ratio=8, pattern=0):
        super(WSAttention, self).__init__()
        self.ca = ChannelAttention(in_planes=in_planes, ratio=ratio, pattern=3)
        # self.eca = ECAAttention()
        self.sa = SpatialAttention(kernel_size)
        self.bn = nn.BatchNorm2d(in_planes)
        self.p = pattern
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, W, H = x.shape
        y = torch.reshape(x, (B, W, C, H))
        if self.p == 0:
            y = y * self.ca(y)
            y = y * self.sa(y)
        elif self.p == 1:
            y = y * self.sa(y)
            y = y * self.ca(y)
        else:
            y0 = y * self.ca(y)
            y1 = y * self.sa(y)
            y = 0.5*(y0 + y1)
        y = self.bn(y)
        out = self.relu(y)
        out = torch.reshape(out, (B, C, W, H))
        return out
class WSAttention1(nn.Module):
    def __init__(self, in_planes=None, kernel_size=7, ratio=8, pattern=0):
        super(WSAttention1, self).__init__()
        # self.ca = ChannelAttention(in_planes=in_planes, ratio=ratio, pattern=3)
        self.eca = ECAAttention()
        self.sa = SpatialAttention(kernel_size)
        self.bn = nn.BatchNorm2d(in_planes)
        self.p = pattern
        self.relu = nn.ReLU()

    def forward(self, x):
        B, C, W, H = x.shape
        y = torch.reshape(x, (B, W, C, H))
        if self.p == 0:
            y = y * self.eca(y)
            y = y * self.sa(y)
        elif self.p == 1:
            y = y * self.sa(y)
            y = y * self.ca(y)
        else:
            y0 = y * self.ca(y)
            y1 = y * self.sa(y)
            y = 0.5*(y0 + y1)
        y = self.bn(y)
        out = self.relu(y)
        out = torch.reshape(out, (B, C, W, H))
        return out

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        # return x*y.expand_as(x)
        return y.expand_as(x)
