""" PhysNet
We repulicate the net pipeline of the orginal paper, but set the input as diffnormalized data.
orginal source:
Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
British Machine Vision Conference (BMVC)} 2019,
By Zitong Yu, 2019/05/05
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019
"""

import math
import pdb

import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
import numpy as np
from torch.autograd import Variable

class DeformConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformConv3d, self).__init__()
        self.kernel_size = kernel_size
        N = kernel_size ** 3
        self.stride = stride
        self.padding = padding
        self.zero_padding = nn.ConstantPad3d(padding, 0)
        self.conv_kernel = nn.Conv3d(in_channels * N, out_channels, kernel_size=1, bias=bias)
        self.offset_conv_kernel = nn.Conv3d(in_channels, N * 3, kernel_size=kernel_size, padding=padding, bias=bias)

        self.mode = "deformable"

    def deformable_mode(self, on=True):  #
        if on:
            self.mode = "deformable"
        else:
            self.mode = "regular"

    def forward(self, x):
        if self.mode == "deformable":
            offset = self.offset_conv_kernel(x)
        else:
            b, c, h, w, d = x.size()
            offset = torch.zeros(b, 3 * self.kernel_size ** 3, h, w, d).to(x)

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 3

        if self.padding:
            x = self.zero_padding(x)

        # (b, 3N, h, w, d)
        p = self._get_p(offset, dtype)
        p = p[:, :, ::self.stride, ::self.stride, ::self.stride]

        # (b, h, w, d, 3N), N == ks ** 3, 3N - 3 coords for each point on the activation map
        p = p.contiguous().permute(0, 2, 3, 4, 1)  # 5D array

        q_sss = Variable(p.data, requires_grad=False).floor()  # point with all smaller coords
        #         q_sss = p.data.floor() - same? / torch.Tensor(p.data).floor()
        q_lll = q_sss + 1  # all larger coords

        # 8 neighbor points with integer coords
        q_sss = torch.cat([
            torch.clamp(q_sss[..., :N], 0, x.size(2) - 1),  # h_coord
            torch.clamp(q_sss[..., N:2 * N], 0, x.size(3) - 1),  # w_coord
            torch.clamp(q_sss[..., 2 * N:], 0, x.size(4) - 1)  # d_coord
        ], dim=-1).long()
        q_lll = torch.cat([
            torch.clamp(q_lll[..., :N], 0, x.size(2) - 1),  # h_coord
            torch.clamp(q_lll[..., N:2 * N], 0, x.size(3) - 1),  # w_coord
            torch.clamp(q_lll[..., 2 * N:], 0, x.size(4) - 1)  # d_coord
        ], dim=-1).long()
        q_ssl = torch.cat([q_sss[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_sls = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_sll = torch.cat([q_sss[..., :N], q_lll[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lss = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_sss[..., 2 * N:]], -1)
        q_lsl = torch.cat([q_lll[..., :N], q_sss[..., N:2 * N], q_lll[..., 2 * N:]], -1)
        q_lls = torch.cat([q_lll[..., :N], q_lll[..., N:2 * N], q_sss[..., 2 * N:]], -1)

        # (b, h, w, d, N)
        mask = torch.cat([
            p[..., :N].lt(self.padding) + p[..., :N].gt(x.size(2) - 1 - self.padding),
            p[..., N:2 * N].lt(self.padding) + p[..., N:2 * N].gt(x.size(3) - 1 - self.padding),
            p[..., 2 * N:].lt(self.padding) + p[..., 2 * N:].gt(x.size(4) - 1 - self.padding),
        ], dim=-1).type_as(p)
        mask = mask.detach()
        floor_p = p - (p - torch.floor(p))  # все еще непонятно, что тут происходит за wtf
        p = p * (1 - mask) + floor_p * mask

        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:2 * N], 0, x.size(3) - 1),
            torch.clamp(p[..., 2 * N:], 0, x.size(4) - 1),
        ], dim=-1)

        # trilinear kernel (b, h, w, d, N)
        g_sss = (1 + (q_sss[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_sss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_sss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lll = (1 - (q_lll[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_lll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_lll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_ssl = (1 + (q_ssl[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_ssl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_ssl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sls = (1 + (q_sls[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_sls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_sls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_sll = (1 + (q_sll[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_sll[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_sll[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lss = (1 - (q_lss[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_lss[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_lss[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lsl = (1 - (q_lsl[..., :N].type_as(p) - p[..., :N])) * (
                    1 + (q_lsl[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 - (q_lsl[..., 2 * N:].type_as(p) - p[..., 2 * N:]))
        g_lls = (1 - (q_lls[..., :N].type_as(p) - p[..., :N])) * (
                    1 - (q_lls[..., N:2 * N].type_as(p) - p[..., N:2 * N])) * (
                            1 + (q_lls[..., 2 * N:].type_as(p) - p[..., 2 * N:]))

        # get values in all 8 neighbor points
        # (b, c, h, w, d, N) - 6D-array
        x_q_sss = self._get_x_q(x, q_sss, N)
        x_q_lll = self._get_x_q(x, q_lll, N)
        x_q_ssl = self._get_x_q(x, q_ssl, N)
        x_q_sls = self._get_x_q(x, q_sls, N)
        x_q_sll = self._get_x_q(x, q_sll, N)
        x_q_lss = self._get_x_q(x, q_lss, N)
        x_q_lsl = self._get_x_q(x, q_lsl, N)
        x_q_lls = self._get_x_q(x, q_lls, N)

        # (b, c, h, w, d, N)
        x_offset = g_sss.unsqueeze(dim=1) * x_q_sss + \
                   g_lll.unsqueeze(dim=1) * x_q_lll + \
                   g_ssl.unsqueeze(dim=1) * x_q_ssl + \
                   g_sls.unsqueeze(dim=1) * x_q_sls + \
                   g_sll.unsqueeze(dim=1) * x_q_sll + \
                   g_lss.unsqueeze(dim=1) * x_q_lss + \
                   g_lsl.unsqueeze(dim=1) * x_q_lsl + \
                   g_lls.unsqueeze(dim=1) * x_q_lls

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv_kernel(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y, p_n_z = np.meshgrid(
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            range(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            indexing='ij')

        # (3N, 1) - 3 coords for each of N offsets
        # (x1, ... xN, y1, ... yN, z1, ... zN)
        p_n = np.concatenate((p_n_x.flatten(), p_n_y.flatten(), p_n_z.flatten()))
        p_n = np.reshape(p_n, (1, 3 * N, 1, 1, 1))
        p_n = torch.from_numpy(p_n).type(dtype)

        return p_n

    @staticmethod
    def _get_p_0(h, w, d, N, dtype):
        p_0_x, p_0_y, p_0_z = np.meshgrid(range(1, h + 1), range(1, w + 1), range(1, d + 1), indexing='ij')
        p_0_x = p_0_x.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_y = p_0_y.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0_z = p_0_z.flatten().reshape(1, 1, h, w, d).repeat(N, axis=1)
        p_0 = np.concatenate((p_0_x, p_0_y, p_0_z), axis=1)
        p_0 = torch.from_numpy(p_0).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w, d = offset.size(1) // 3, offset.size(2), offset.size(3), offset.size(4)

        # (1, 3N, 1, 1, 1)
        p_n = self._get_p_n(N, dtype).to(offset)
        # (1, 3N, h, w, d)
        p_0 = self._get_p_0(h, w, d, N, dtype).to(offset)
        p = p_0 + p_n + offset

        return p

    def _get_x_q(self, x, q, N):
        b, h, w, d, _ = q.size()

        #           (0, 1, 2, 3, 4)
        # x.size == (b, c, h, w, d)
        padded_w = x.size(3)
        padded_d = x.size(4)
        c = x.size(1)
        # (b, c, h*w*d)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, d, N)
        # offset_x * w * d + offset_y * d + offset_z
        index = q[..., :N] * padded_w * padded_d + q[..., N:2 * N] * padded_d + q[..., 2 * N:]
        # (b, c, h*w*d*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, d, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, d, N = x_offset.size()
        x_offset = x_offset.permute(0, 1, 5, 2, 3, 4)
        x_offset = x_offset.contiguous().view(b, c * N, h, w, d)

        return x_offset


def deform_conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return DeformConv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PhysNet_padding_Encoder_Decoder_MAX_def(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_padding_Encoder_Decoder_MAX_def, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            #DeformConv3d(3, 16, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            #nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            DeformConv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            #nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            DeformConv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        # x [32, T/2, 32,32]    Temporal halve
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        # x [64, T, 1,1]    -->  groundtruth left and right - 7
        x = self.poolspa(x)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG = x.view(-1, length)

        return rPPG, x_visual, x_visual3232, x_visual1616
