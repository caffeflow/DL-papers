import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# upscale the spatical size by a factor of 2


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True)
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )
        self.relu = nn.ReLU(True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class CA_NET(nn.Module):
    # modified from vae epamples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.c_dim = cfg.TEXT.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim*2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        # 用一个全连接去学习分布的logvar和mu
        mu = x[:, :self.c_dim]
        logvar = x[:self.c_dim:]
        return mu, logvar

    def reparametirze(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul_(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametirze(mu, logvar)
        return c_code, mu, logvar


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            # stage2
            self.outlogits = nn.Sequential(
                conv3x3(ndf*8 + nef, ndf*8),
                nn.BatchNorm2d(ndf*8),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf*8, 1, kernel_size=4, stride=4),
                nn.Sigmoid()
            )
        else:
            # stage1
            self.outlogits = nn.Sequential(
                nn.Conv2d(ndf*8, 1, kernel_size=4, stride=4),
                nn.Sigmoid()
            )

    def forward(self, h_code, c_code=None):
        # conditioning output
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size(ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


# ######## stageI GAN ##########3
class STAGE1_G(nn.Module):
    def __init__(self):
        super(STAGE1_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM * 8
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_dim
        self.define_module()

    def define_module(self):
        ninput = self.z_dim + self.ef_dim
        ngf = self.gf_dim
        self.ca_net = CA_NET()

        self.upsample1 = upBlock(ngf, ngf//2)
        self.upsample2 = upBlock(ngf//2, ngf//4)
        self.upsample3 = upBlock(ngf//4, ngf//8)
        self.upsample4 = upBlock(ngf//8, ngf//16)
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh()
        )

    def forward(self, text_embedding, noise):
        c_code, mu, logvar = self.ca_net(text_embedding)
        z_c_code = torch.cat((noise, c_code), 1)  # 条件增强
        h_code = self.fc(z_c_code)

        h_code = h_code.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        fake_img = self.img(h_code)
        return None, fake_img, mu, logvar

    class STAGE1_D(nn.Module):
        def __init__(self):
            super(STAGE1_D, self).__init__()
            self.df_dim = cfg.GAN.DF_DIM
            self.ef_dim = cfg.GAN.CONDITION_DIM
            self.define_module()

        def define_module(self):
            ndf, nef = self.df_dim, self.ef_dim
            self.encode_img = nn.Sequential(
                nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size (ndf*2) x 16 x 16
                nn.Conv2d(ndf*2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size (ndf*4) x 8 x 8
                nn.Conv2d(ndf*4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                # state size (ndf * 8) x 4 x 4)
                nn.LeakyReLU(0.2, inplace=True)
            )

            self.get_cond_logits = D_GET_LOGITS(ndf, nef)
            self.get_uncond_logits = None

        def forward(self, image):
            img_embedding = self.encode_img(image)
            return img_embedding

# ############### stageII GAN ###############


class STAGE2_G(nn.Module):
    def __init__(self, STAGE1_G):
        super(STAGE2_G, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.ef_dim = cfg.GAN.CONDITION_DIM
        self.z_dim = cfg.Z_DIM
        self.STAGE1_G = STAGE1_G
        # fix parameters of stageI GAN
        for param in self.STAGE1_G.parameters:
            param.requires_grad = False
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block())
