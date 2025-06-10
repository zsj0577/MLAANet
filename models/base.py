import torch
import torch.nn as nn
from torch.nn import Softmax
# from inplace_abn import InPlaceABN, InPlaceABNSync
BatchNorm2d = nn.BatchNorm2d#SyncBN#functools.partial(InPlaceABNSync, activation='identity')

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
            ConvBlock(self.out_channels, self.out_channels),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.double_conv_block(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # Up Conv block
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(self.in_channels, self.out_channels),

        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.up_conv(x)
        return x


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class RCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCCAModule, self).__init__()
        inter_channels = in_channels//2
        # inter_channels = out_channels
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(inter_channels), nn.ReLU(inplace=False))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(inter_channels), nn.ReLU(inplace=False))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(out_channels), nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
            )

    def forward(self, x, recurrence=1):
        output = self.conva(x)
        for i in range(recurrence):
            output = self.cca(output)
        output = self.convb(output)

        output = self.bottleneck(torch.cat([x, output], 1))
        return output


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x


class AttentionBlock(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out
# class AttentionBlock(nn.Module):
#
#     def __init__(self, F_g, F_l, F_int):
#         super(AttentionBlock, self).__init__()
#
#         self.W_g = nn.Sequential(
#             # nn.Conv2d(F_l, F_int, kernel_size=1,
#             #           stride=1, padding=0, bias=True),
#             nn.Conv2d(F_l, F_int, (1, 5),
#                       stride=1, padding=(0, 2), bias=True),
#             nn.Conv2d(F_int, F_int, (5, 1) ,
#                       stride=1, padding=(2, 0), bias=True),
#             nn.BatchNorm2d(F_int),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(F_int, F_int, 1,
#                      stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int),
#
#
#             nn.Conv2d(F_int, F_int, (1, 5),
#                       stride=1, padding=(0, 2), bias=True),
#             nn.Conv2d(F_int, F_int, (5, 1),
#                       stride=1, padding=(2, 0), bias=True),
#             nn.BatchNorm2d(F_int),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(F_int, F_int, 1,
#                       stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#
#         )
#
#         self.W_x = nn.Sequential(
#             # nn.Conv2d(F_g, F_int, kernel_size=1,
#             #           stride=1, padding=0, bias=True),
#             nn.Conv2d(F_g, F_int, (1, 5),
#                       stride=1, padding=(0, 2), bias=True),
#             nn.Conv2d(F_int, F_int, (5, 1),
#                       stride=1, padding=(2, 0), bias=True),
#             nn.BatchNorm2d(F_int),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(F_int, F_int, 1,
#                       stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int),
#
#             nn.Conv2d(F_int, F_int, (1, 5),
#                       stride=1, padding=(0, 2), bias=True),
#             nn.Conv2d(F_int, F_int, (5, 1),
#                       stride=1, padding=(2, 0), bias=True),
#             nn.BatchNorm2d(F_int),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(F_int, F_int, 1,
#                       stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#
#
#         )
#         self.W_g2 = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1,
#                       stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#         self.W_x2 = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1,
#                       stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(F_int)
#         )
#
#
#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.psi2 = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.psi_final_fun = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#
#         # g1 = self.W_g(g)
#         # x1 = self.W_x(x)
#
#
#
#         # psi = self.relu(g1 + x1)
#         psi = self.relu(g1 + x1)
#         psi_final = self.psi(psi)
#         # pi = torch.cat((psi, psi2), dim=1)
#         # psi_final = self.psi_final_fun(pi)
#
#         # psi = self.psi(psi)
#         #
#         # psi2 = self.relu(g2+ x2)
#         # psi2 = self.psi2(psi2)
#         # psi_final = 0.5*psi+0.5*psi2
#         out = x * psi_final
#         return out


class NestedBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(NestedBlock, self).__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels

        self.nest_block = nn.Sequential(
            ConvBlock(self.in_channels, self.mid_channels),
            ConvBlock(self.mid_channels, self.out_channels),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        output = self.nest_block(x)
        return output


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    image = torch.rand((1, 3, 522, 775)).to(DEVICE)

    model = DoubleConvBlock(3, 1).to(DEVICE)
    res = model(image)
    print(res.shape)
