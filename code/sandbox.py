# this file is for experiments like checking tensor sizes and stuff
import os
import random
from math import log2

import cv2
import matplotlib.pyplot as plt
import torchvision
from torch import nn
from torchvision.utils import save_image

sandbox_backend = plt.get_backend()
import numpy as np
import torch
from torch.nn import ConvTranspose2d, Sequential
from torch.nn import functional as F

from Datasets.utils import visflow

plt.switch_backend(sandbox_backend)




class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x


class DiscreteVAE(nn.Module):
    def __init__(
            self,
            image_size=256,
            num_tokens=512,
            codebook_dim=512,
            num_layers=3,
            num_resnet_blocks=0,
            hidden_dim=64,
            channels=3,
            smooth_l1_loss=False,
            temperature=0.9,
            straight_through=False,
            kl_div_loss_weight=0.,
            normalization=((*((0.5,) * 3), 0), (*((0.5,) * 3), 1))
    ):
        super().__init__()
        assert log2(image_size).is_integer(), 'image size must be a power of 2'
        assert num_layers >= 1, 'number of layers must be greater than or equal to 1'
        has_resblocks = num_resnet_blocks > 0

        self.channels = channels
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        hdim = hidden_dim

        enc_chans = [hidden_dim] * num_layers
        dec_chans = list(reversed(enc_chans))

        enc_chans = [channels, *enc_chans]

        dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

        enc_layers = []
        dec_layers = []

        for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
            enc_layers.append(nn.Sequential(nn.Conv2d(enc_in, enc_out, 4, stride=2, padding=1), nn.ReLU()))
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(dec_in, dec_out, 4, stride=2, padding=1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))
            enc_layers.append(ResBlock(enc_chans[-1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv2d(codebook_dim, dec_chans[1], 1))

        enc_layers.append(nn.Conv2d(enc_chans[-1], num_tokens, 1))
        dec_layers.append(nn.Conv2d(dec_chans[-1], channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.kl_div_loss_weight = kl_div_loss_weight

        # take care of normalization within class
        self.normalization = tuple(map(lambda t: t[:channels], normalization))

    #     self._register_external_parameters()
    #
    # def _register_external_parameters(self):
    #     """Register external parameters for DeepSpeed partitioning."""
    #     if (
    #             not distributed_utils.is_distributed
    #             or not distributed_utils.using_backend(
    #                 distributed_utils.DeepSpeedBackend)
    #     ):
    #         return
    #
    #     deepspeed = distributed_utils.backend.backend_module
    #     deepspeed.zero.register_external_parameter(self, self.codebook.weight)

    def norm(self, images):
        if not exists(self.normalization):
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        means, stds = map(lambda t: rearrange(t, 'c -> () c () ()'), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        logits = self(images, return_logits=True)
        codebook_indices = logits.argmax(dim=1).flatten(1)
        return codebook_indices

    def decode(
            self,
            img_seq
    ):
        image_embeds = self.codebook(img_seq)
        b, n, d = image_embeds.shape
        h = w = int(sqrt(n))

        image_embeds = rearrange(image_embeds, 'b (h w) d -> b d h w', h=h, w=w)
        images = self.decoder(image_embeds)
        return images

    def forward(
            self,
            img,
            return_loss=False,
            return_recons=False,
            return_logits=False,
            temp=None
    ):
        device, num_tokens, image_size, kl_div_loss_weight = img.device, self.num_tokens, self.image_size, self.kl_div_loss_weight
        assert img.shape[-1] == image_size and img.shape[
            -2] == image_size, f'input must have the correct image size {image_size}'

        img = self.norm(img)

        logits = self.encoder(img)

        if return_logits:
            return logits  # return logits for getting hard image indices for DALL-E training

        temp = default(temp, self.temperature)
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=self.straight_through)
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        out = self.decoder(sampled)

        if not return_loss:
            return out

        # reconstruction loss

        recon_loss = self.loss_fn(img, out)

        # kl divergence

        logits = rearrange(logits, 'b n h w -> b (h w) n')
        log_qy = F.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1. / num_tokens], device=device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, 'batchmean', log_target=True)

        loss = recon_loss + (kl_div * kl_div_loss_weight)

        if not return_recons:
            return loss

        return loss, out



def decoder_from_vae_ceckpoint(checkpoint_path='discrete_vae.pth'):
    latent_dim = 3
    vae = DiscreteVAE(
        image_size=512,
        num_layers=3,  # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
        num_tokens=latent_dim,
        # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
        codebook_dim=latent_dim,  # codebook dimension
        hidden_dim=64,  # hidden dimension
        num_resnet_blocks=1,  # number of resnet blocks
        temperature=0.9,  # gumbel softmax temperature, the lower this is, the harder the discretization
        straight_through=False,  # straight-through for gumbel softmax. unclear if it is better one way or the other
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')
    vae.load_state_dict(torch.load(checkpoint_path, map_location=device))
    dec = vae.decoder
    enc = vae.encoder
    #
    latim = enc(torch.randn(1, 3, 512, 512))
    seed = torch.rand(1, latent_dim, 56, 80)
    scale=1.0
    randim = dec(latim)
    print(randim.shape)
    rand_im = dec(scale * seed)[0]
    print(rand_im.shape)
    return dec





class MyEncoder(torch.nn.Module):
    def __init__(self, out_channels):
        super(MyEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, (70, 100), 8, (1, 4))
        self.conv2 = nn.Conv2d(10, 30, 4, 1, 1)
        self.conv3 = nn.Conv2d(30, out_channels, (35, 50), 3, (4, 5))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return torch.clamp(x, min=0, max=2)


def random_initiation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = AdjEncoder(out_channels=3).to(device)
    kernel_shape = encoder.output_shape
    print(f'kernel shape: {kernel_shape}')
    model = AdjDecoder(in_channels=3, kernel_shape=kernel_shape).to(device)
    return model


kernel_sizes = 10 * [(3, 3)]
inner_channels = [200, 200, 150, 150, 100, 100, 50, 50, 25, 25, 3, 3]
strides = 5 * [1, 2]


class AdjEncoder(torch.nn.Module):
    # this class' whole purpose is to calculate the kernel shape
    def __init__(self, out_channels):
        super(AdjEncoder, self).__init__()
        layers = []
        prev_channel = 3
        for i, filter in enumerate(kernel_sizes):
            if i < len(kernel_sizes) - 1:
                current_channel = inner_channels[i]
            else:
                current_channel = out_channels
            layers.append(nn.Conv2d(prev_channel, current_channel, filter, 1, 0))
            layers.append(nn.ReLU())
            prev_channel = current_channel
        self.cnn = nn.Sequential(*layers)
        temp = torch.randn(1, 3, 448, 640)
        self.output_shape = self.forward(temp).shape

    def forward(self, x):
        return torch.clamp(self.cnn(x), min=0, max=2)


class AdjDecoder(torch.nn.Module):
    def __init__(self, in_channels, kernel_shape, kernel_value=100):
        super(AdjDecoder, self).__init__()
        layers = []
        prev_channel = in_channels
        for i, filter in enumerate(reversed(kernel_sizes)):  # reversed??
            if i < len(kernel_sizes) - 1:
                current_channel = inner_channels[i]
            else:
                current_channel = 3
            layers.append(nn.ConvTranspose2d(prev_channel, current_channel, filter, 1, 0))
            layers.append(nn.ReLU(inplace=True))
            prev_channel = current_channel
        self.cnn = nn.Sequential(*layers)
        self.kernel_shape = kernel_shape
        self.kernel_value = kernel_value

    def forward(self, x):
        return self.cnn(x)
        return torch.clamp(self.cnn(x), 0, 255)

    def sample(self, device=None, seed=None):
        device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if seed is None:
            pert = self.forward(self.kernel_value * torch.ones(self.kernel_shape).to(device))
        else:
            pert = self.forward(seed)
        return pert
        return torch.clamp(pert, 0, 255)


class PertGenerator(torch.nn.Module):
    '''
    this code is not clean, nor modular.
    '''

    def __init__(self, h_dim=20):
        super(PertGenerator, self).__init__()
        # self.kernel_size = 4
        # self.kernel = torch.ones(1, 1, kernel_size, kernel_size)
        # self.kernel = torch.tensor([[[[1, 0, 1, 0],
        #                     [1, 0, 1, 1],
        #                     [1, 1, 0, 0],
        #                     [0, 0, 0, 1]]]], dtype=torch.float32)

        # self.kernel = torch.rand(1, 1, 224, 320)
        # c1 = ConvTranspose2d(1, 1, kernel_size=(6, 6), stride=(4, 4), padding=2)
        # c2 = ConvTranspose2d(1, 2, kernel_size=(13, 19), stride=(5, 5), padding=2)
        # c3 = ConvTranspose2d(2, 3, kernel_size=(14, 12), stride=(6, 8), padding=2)
        #
        self.kernel = torch.ones(1, h_dim, 7, 10)
        # c1 = ConvTranspose2d(1, 1, kernel_size=(70, 100), stride=(1, 1), padding=0)
        # c2 = ConvTranspose2d(1, 2, kernel_size=(70, 100), stride=(1, 1), padding=0)
        # c3 = ConvTranspose2d(2, 3, kernel_size=(35, 47), stride=(3, 3), padding=1, output_padding=1)
        #
        # self.cnn = Sequential(c1, c2, c3)

        ''' 
        # best so far:
        Average target_rms_crit_adv_delta = 0.11246040026657284
        Average mean_partial_rms_crit_adv_delta = 0.4526335008442402     
        '''
        output_size = self.kernel.flatten().shape[0]
        self.fc = nn.Linear(output_size,
                            output_size)  # c1 = ConvTranspose2d(1, 20, kernel_size=(1, 1), stride=(1, 1), padding=(0, 1))
        self.c2 = ConvTranspose2d(h_dim, 30, kernel_size=(14, 25), stride=(2, 2), padding=(1, 4))
        self.c3 = ConvTranspose2d(30, 20, kernel_size=(35, 50), stride=(3, 3), padding=(4, 7))
        self.c4 = ConvTranspose2d(20, 3, kernel_size=(70, 100), stride=(4, 4), padding=(1, 4))
        # self.cnn = Sequential(c2, c3, c4)

        # c1 = ConvTranspose2d(1, 3, kernel_size=(448, 640), stride=(1, 1), padding=(0, 0))
        # self.cnn = Sequential(c1)

        # self.kernel = torch.ones(1, h_dim, 7, 10)
        # self.fc = nn.Linear(h_dim * 7 * 10, h_dim * 7 * 10)
        # channels = [20, 50, 10, 30, 10, 3]
        # up_layers = []
        # up_layers.append(nn.ConvTranspose2d(h_dim, channels[0], kernel_size=4, stride=2, padding=1))
        # print(up_layers[-1](self.kernel).shape)
        # for i in range(5):
        #     # up_layers.append(ConvTranspose2d(channels[i], channels[i], kernel_size=3, stride=1, padding=1, dilation=1))
        #     up_layers.append(torch.nn.ReLU())  # without its 102 and with its 103
        #     up_layers.append(ConvTranspose2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, dilation=1,
        #                         output_padding=1))
        # self.cnn = Sequential(*up_layers)

    def sample(self, device, seed=None):
        if seed is None:
            seed = self.kernel
        x = self.fc(seed.flatten().to(device))
        x = x.reshape(seed.shape)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        pert = x
        return torch.clamp(pert, 0, 255)


class PertGeneratorT(torch.nn.Module):
    '''
    this code is not clean, nor modular.
    '''

    def __init__(self):
        super(PertGeneratorT, self).__init__()
        self.kernel = torch.nn.Parameter(torch.randn(1, 200, 14, 20), requires_grad=False)
        self.factor = torch.nn.Parameter(100.0 * torch.ones(1), requires_grad=False)
        channels = [200, 150, 100, 50, 25, 3]
        up_layers = []
        for i in range(5):
            up_layers.append(
                ConvTranspose2d(channels[i], channels[i], kernel_size=3, stride=1, padding=1, dilation=1))
            # up_layers.append(torch.nn.ReLU())  # without its 102 and with its 103
            up_layers.append(torch.nn.Dropout(0.1))
            up_layers.append(
                ConvTranspose2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, dilation=1,
                                output_padding=1))
        self.cnn = Sequential(*up_layers)

    def sample(self, device):
        return torch.clamp(self.cnn(self.factor * self.kernel.to(device)), 0, 1)

    def forward(self, seed=None):
        if seed is None:
            seed = self.kernel
        x = self.cnn(self.factor * seed)
        return torch.clamp(x, 0, 1)


class LinConvGen(nn.Module):

    def linear(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.ReLU(inplace=True)
        )

    def __init__(self, ):
        super(LinConvGen, self).__init__()

        fc1 = self.linear(3, 32)
        fc2 = self.linear(32, 128)
        fc3 = nn.Linear(128, 256 * 6)
        lk = 7
        rk = 10

        conv_1 = nn.ConvTranspose2d(256, 256, kernel_size=(lk, rk), stride=1, padding=0)
        conv1 = nn.ConvTranspose2d(256, 128, kernel_size=(lk, rk), stride=1, padding=0)
        conv_2 = nn.ConvTranspose2d(128, 128, kernel_size=(lk, rk), stride=1, padding=0)
        conv2 = nn.ConvTranspose2d(128, 128, kernel_size=(lk, rk), stride=1, padding=0)
        conv_3 = nn.ConvTranspose2d(128, 128, kernel_size=(lk, rk), stride=1, padding=0)
        conv3 = nn.ConvTranspose2d(128, 64, kernel_size=(lk, rk), stride=1, padding=0)
        conv_4 = nn.ConvTranspose2d(64, 64, kernel_size=(lk, rk), stride=1, padding=0)
        conv4 = nn.ConvTranspose2d(64, 32, kernel_size=(lk, rk), stride=1, padding=0)
        conv_5 = nn.ConvTranspose2d(32, 32, kernel_size=(lk, rk), stride=1, padding=0)
        conv5 = nn.ConvTranspose2d(32, 140, kernel_size=(lk, rk), stride=1, padding=0, )

        self.seed = torch.randn(3)
        self.output_shape = (3, 448, 640)
        self.sequential = torch.nn.Sequential(fc1, fc2, fc3)
        # self.cnn = torch.nn.Sequential(conv1, conv2, conv3, conv4, conv5)
        self.cnn = torch.nn.Sequential(conv_1, conv1, conv_2, conv2, conv_3, conv3, conv_4, conv4, conv_5, conv5)

    def sample(self, device, seed=None):
        return self.forward(self.seed)

    def forward(self, x):
        output = self.sequential(x.flatten())
        output = output.view((1, 256, 2, 3))
        x = self.cnn(output)
        x = x.view(1, -1)[:, :-840]
        # print('xshape', x.shape)
        num_elem = x.view(1, -1).shape[1]
        # print(num_elem % 448)
        # print(num_elem % 640)
        x = x.view(3, 448, -1)
        return x


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def reversed_conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
    if bn_layer:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation,
                               bias=bias, output_padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride,
                               dilation=dilation, output_padding=0),
            nn.ReLU(inplace=True)
        )


def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes),
        nn.ReLU(inplace=True)
    )


class ReversedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(ReversedBasicBlock, self).__init__()

        temp = planes
        planes = inplanes
        inplanes = temp

        self.conv1 = reversed_conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.ConvTranspose2d(planes, planes, 3, 1, pad, dilation=dilation, output_padding=0)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        # print(out.shape, x.shape)
        # out += x

        return F.relu(out, inplace=True)


class ReversedVOFlowRes(nn.Module):
    def __init__(self):
        super().__init__()
        inputnum = 4
        blocknums = [2, 2, 3, 4, 6, 7, 3]
        outputnums = [32, 64, 64, 128, 128, 256, 256]

        self.firstconv = nn.Sequential(reversed_conv(32, 32, 3, 1, 1, 1),
                                       reversed_conv(32, 32, 3, 1, 1, 1),
                                       reversed_conv(32, inputnum, 3, 2, 1, 1, False)
                                       )

        self.inplanes = 32

        self.layer1 = self._make_layer(ReversedBasicBlock, outputnums[2], blocknums[2], 2, 1, 1)  # 40 x 28
        self.layer2 = self._make_layer(ReversedBasicBlock, outputnums[3], blocknums[3], 2, 1, 1)  # 20 x 14
        self.layer3 = self._make_layer(ReversedBasicBlock, outputnums[4], blocknums[4], 2, 1, 1)  # 10 x 7
        self.layer4 = self._make_layer(ReversedBasicBlock, outputnums[5], blocknums[5], 2, 1, 1)  # 5 x 4
        self.layer5 = self._make_layer(ReversedBasicBlock, outputnums[6], blocknums[6], 2, 1, 1)  # 3 x 2
        fcnum = outputnums[6] * 6

        fc1_trans = linear(fcnum, 128)
        fc2_trans = linear(128, 32)
        fc3_trans = nn.Linear(32, 3)

        fc1_rot = linear(fcnum, 128)
        fc2_rot = linear(128, 32)
        fc3_rot = nn.Linear(32, 3)

        self.reversed_fc1 = nn.Linear(3, 32)
        self.reversed_fc2 = linear(32, 128)
        self.reversed_fc3 = nn.Linear(128, fcnum)
        # self.reversed_fc = nn.Sequential(reversed_fc1, reversed_fc2, reversed_fc3)

        self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.ConvTranspose2d(planes * block.expansion, self.inplanes,
                                            kernel_size=1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*reversed(layers))

    def forward(self, x):
        batchsize = x.shape[0]
        x = self.reversed_fc1(x)
        x = self.reversed_fc2(x)
        x = self.reversed_fc3(x)
        num_elem = x.view(1, -1).shape[1]
        width = int(np.sqrt(num_elem // 256))
        x = x.view(batchsize, 256, width, -1)
        x = self.layer5(x)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.firstconv(x)

        return x
        # x_trans = self.voflow_trans(x)
        # x_rot = self.voflow_rot(x)
        # return torch.cat((x_trans, x_rot), dim=1)


def reversed_deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Conv2d(out_planes, in_planes, kernel_size, stride, padding, bias=True)


def reversed_predict_flow(in_planes):
    return nn.ConvTranspose2d(2, in_planes, kernel_size=3, stride=1, padding=1, bias=True)


class Flow2ImgExpander(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = reversed_conv(2, 16, kernel_size=3, stride=2)
        self.conv1aa = reversed_conv(16, 16, kernel_size=3, stride=1)
        self.conv1b = reversed_conv(16, 16, kernel_size=3, stride=1)
        self.conv2a = reversed_conv(16, 32, kernel_size=3, stride=2)
        self.conv2aa = reversed_deconv(32, 32, kernel_size=3, stride=1)
        self.conv2b = reversed_conv(32, 32, kernel_size=3, stride=1)
        self.conv3a = reversed_conv(32, 64, kernel_size=3, stride=2)
        self.conv3aa = reversed_conv(64, 64, kernel_size=3, stride=1)
        self.conv3b = reversed_deconv(64, 64, kernel_size=3, stride=1)
        self.conv4a = reversed_conv(64, 32, kernel_size=3, stride=2)
        self.conv4aa = reversed_deconv(32, 32, kernel_size=3, stride=1)
        self.conv4b = reversed_conv(32, 32, kernel_size=3, stride=1)
        self.conv5a = reversed_conv(32, 16, kernel_size=3, stride=2)
        self.conv5aa = reversed_conv(16, 16, kernel_size=3, stride=1)
        self.conv5b = reversed_conv(16, 16, kernel_size=3, stride=1)
        self.conv6aa = reversed_conv(16, 8, kernel_size=3, stride=2)
        self.conv6a = reversed_conv(8, 8, kernel_size=3, stride=1)
        self.conv6b = reversed_conv(8, 8, kernel_size=3, stride=1)

        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)

        # self.corr    = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1, corr_multiply=1)
        self.leakyRELU = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = x[:, [0, 1], :, :]
        x2 = x[:, [2, 3], :, :]
        c11 = self.conv1b(self.conv1aa(self.conv1a(x1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(x2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c13 = c13[:, :, :448, :640]
        c23 = c23[:, :, :448, :640]
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c14 = c14[:, :, :448, :640]
        c24 = c24[:, :, :448, :640]
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c15 = c15[:, :, :448, :640]
        c25 = c25[:, :, :448, :640]
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))
        c16 = c16[:, :, :448, :640]
        c26 = c26[:, :, :448, :640]
        x = torch.cat((c16, c26), 1)
        x = self.final_conv(x)
        x = x
        return x[:, :, :448, :640]


class FlowGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.vflow = ReversedVOFlowRes()
        self.expander = Flow2ImgExpander()
        self.seed = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=False)
        # initialize weights of the network
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.ConvTranspose2d):
                # print(m.weight.data.mean())
                factor = 2 + (i // 10) % 4
                print(factor)
                m.weight.data *= factor

    def forward(self, x):
        x = self.vflow(x)
        x = self.expander(x)
        return x

    def sample(self, device=None, seed=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if seed is None:
            seed = self.seed
        return self.forward(seed.to(device))


def find_multiplied_numbers(target_num, array):
    def dot_prod(arr_1, arr_2):
        return sum(map(lambda x, y: x * y, arr_1, arr_2))

    base = [int(target_num / variable) for variable in array]
    result = [0] * len(array)
    while dot_prod(result, array) != target_num:
        result[0] += 1
        for i in range(1, len(result)):
            if result[i] == base[i]:
                result[i] = 0
                result[i - 1] += 1
    return f"For the array {array}, the multiply array: {result}"


def reversed_experiment():
    print('making voflo')
    voflo = ReversedVOFlowRes()
    print(f' voflo has {get_n_params(voflo)} parameters')
    print('making expander')
    expander = Flow2ImgExpander()
    print(f' expander has {get_n_params(expander)} parameters')
    print('making input')
    model = nn.Sequential(voflo, expander)
    print(f' model has {get_n_params(model)} parameters')
    modelf = FlowGen()
    print(f' modelf has {get_n_params(modelf)} parameters')
    input = torch.randn(1, 3)
    print('running model')
    output = model(input)
    print('output shape', output.shape)
    plt.imsave('experiments/gen1.png', torch.clamp(output[0].detach().permute(1, 2, 0), 0, 1).numpy())
    imshow(output[0])


def experiment_tartan():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
        if bn_layer:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride,
                          dilation=dilation, bias=bias),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride,
                          dilation=dilation),
                nn.ReLU(inplace=True)
            )

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
            super(BasicBlock, self).__init__()

            self.conv1 = conv(inplanes, planes, 3, stride, pad, dilation)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)

            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            out = self.conv1(x)
            out = self.conv2(out)

            if self.downsample is not None:
                x = self.downsample(x)
            out += x

            return F.relu(out, inplace=True)

    class VOFlowRes(nn.Module):
        def __init__(self):
            super(VOFlowRes, self).__init__()
            inputnum = 4
            blocknums = [2, 2, 3, 4, 6, 7, 3]
            outputnums = [32, 64, 64, 128, 128, 256, 256]

            self.firstconv = nn.Sequential(conv(inputnum, 32, 3, 2, 1, 1, False),
                                           conv(32, 32, 3, 1, 1, 1),
                                           conv(32, 32, 3, 1, 1, 1))

            self.inplanes = 32

            self.layer1 = self._make_layer(BasicBlock, outputnums[2], blocknums[2], 2, 1, 1)  # 40 x 28
            self.layer2 = self._make_layer(BasicBlock, outputnums[3], blocknums[3], 2, 1, 1)  # 20 x 14
            self.layer3 = self._make_layer(BasicBlock, outputnums[4], blocknums[4], 2, 1, 1)  # 10 x 7
            self.layer4 = self._make_layer(BasicBlock, outputnums[5], blocknums[5], 2, 1, 1)  # 5 x 4
            self.layer5 = self._make_layer(BasicBlock, outputnums[6], blocknums[6], 2, 1, 1)  # 3 x 2
            fcnum = outputnums[6] * 6

            fc1_trans = linear(fcnum, 128)
            fc2_trans = linear(128, 32)
            fc3_trans = nn.Linear(32, 3)

            fc1_rot = linear(fcnum, 128)
            fc2_rot = linear(128, 32)
            fc3_rot = nn.Linear(32, 3)

            self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
            self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)

        def _make_layer(self, block, planes, blocks, stride, pad, dilation):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.ConvTranspose2d(self.inplanes, planes * block.expansion,
                                                kernel_size=1, stride=stride)

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

            return nn.Sequential(layers)

        def forward(self, x):
            print(f'inside forward, x.shape: {x.shape}')
            x = self.firstconv(x)
            print(f'after firstconv, x.shape: {x.shape}')
            x = self.layer1(x)
            print(f'after layer1, x.shape: {x.shape}')
            x = self.layer2(x)
            print(f'after layer2, x.shape: {x.shape}')
            x = self.layer3(x)
            print(f'after layer3, x.shape: {x.shape}')
            x = self.layer4(x)
            print(f'after layer4, x.shape: {x.shape}')
            x = self.layer5(x)
            print(f'after layer5, x.shape: {x.shape}')
            return x
            x = x.view(x.shape[0], -1)
            x_trans = self.voflow_trans(x)
            x_rot = self.voflow_rot(x)
            return torch.cat((x_trans, x_rot), dim=1)

    model = VOFlowRes()
    flow = torch.randn(7, 4, 112, 160)
    print(model(flow).shape)


def experiment_conv_sizes():
    #### desired shape : [1, 3, 448, 640]

    input = torch.randn(1, 3, 448, 640)
    layers = []
    layers.append(nn.Conv2d(3, 1, (70, 100), 1, 0))
    temp = layers[0](input)
    print(f'temp: {temp.shape}')
    # o2 = ConvTranspose2d(1, 1, (70, 100), 1, 0)(temp)
    # print(f'o2.shape: {o2.shape}')
    layers.append(nn.Conv2d(1, 1, (70, 100), 1, 0))
    temp = layers[1](temp)
    print(temp.shape)
    layers.append(nn.Conv2d(1, 1, (70, 100), 1, 0))
    temp = layers[1](temp)
    print(temp.shape)
    layers.append(nn.Conv2d(1, 1, (70, 100), 1, 0))
    temp = layers[1](temp)
    print(temp.shape)
    layers.append(nn.Conv2d(1, 1, (70, 100), 1, 0))
    temp = layers[1](temp)
    print(temp.shape)
    layers.append(nn.Conv2d(1, 1, (70, 100), 1, 0))

    # conv3 = nn.Conv2d(1, 1, (14, 20), 2, (1,2))

    encoder = nn.Sequential(*layers)
    print(encoder(input).shape)

    dlayers = 5 * [nn.ConvTranspose2d(1, 1, (70, 100), 1, 0)]
    decoder = nn.Sequential(*dlayers)
    print(decoder(temp).shape)
    return

    # kernel = torch.ones(1, 1, 224, 320)
    # c1 = ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=2)
    # c2 = ConvTranspose2d(1, 2, kernel_size=(4, 4), stride=(1, 1), padding=1)
    # c3 = ConvTranspose2d(2, 3, kernel_size=(6, 6), stride=(2, 2), padding=1)
    # model = Sequential(c1, c2, c3)

    kernel = torch.ones(1, 1, 1, 1)
    # c1 = ConvTranspose2d(1, 20, kernel_size=(7, 10), stride=(1, 1), padding=(0, 1))
    # c2 = ConvTranspose2d(20, 30, kernel_size=(14, 25), stride=(2, 2), padding=(1, 2))
    # c3 = ConvTranspose2d(30, 20, kernel_size=(35, 50), stride=(3, 3), padding=(4, 7))
    # c4 = ConvTranspose2d(20, 3, kernel_size=(70, 100), stride=(8, 8), padding=(1, 4))
    # for i, layer in enumerate([c1, c2, c3, c4]):
    #     layer.weight.data.fill_(0.01/(i+1))
    # model = Sequential(c1, c2, c3, c4)

    # c1 = ConvTranspose2d(1, 3, kernel_size=(448, 640), stride=(1, 1), padding=(0, 0))
    # model = Sequential(c1)
    kernel = torch.randn(1, 200, 14, 20)
    up_layers = []
    channels = [200, 150, 100, 50, 25, 3]
    # channels = [1, 500, 1000, 2000, 700, 3]
    print(channels)
    for i in range(5):
        up_layers.append(ConvTranspose2d(channels[i], channels[i], kernel_size=3, stride=1, padding=1, dilation=1))
        # up_layers.append(nn.ReLU())
        up_layers.append(ConvTranspose2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, dilation=1,
                                         output_padding=1))
        # up_layers.append(ConvTranspose2d(1, 1, kernel_size=1, stride=2, padding=1))
        temp_model = Sequential(*up_layers)
        output = temp_model(kernel)
        print(f'{i} - output.shape: {output.shape}')
        # up_layers.append(ConvTranspose2d(1, 1, kernel_size=1, stride=2))

    print(f'len up_layers: {len(up_layers)}')

    # for i in range(3):
    #     up_layers.append(ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1))
    #     up_layers.append(ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, dilation=1))

    down_layers = []
    # for i in range(3):
    #     down_layers.append(ConvTranspose2d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1))
    #     down_layers.append(ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, dilation=1))
    model = Sequential(*up_layers, *down_layers)

    print(f'{get_n_params(model)} <? {3 * 448 * 640}')

    output = model(kernel)[0]
    print(output.shape)
    plt.imshow(output.detach().permute(1, 2, 0))
    plt.show()


def experiment_num_params():
    kernel = torch.tensor([[[[1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1],
                             [1, 1, 1, 1]]]], dtype=torch.float32)
    print(kernel.shape)
    model = PertGenerator()
    print(get_n_params(model))
    print(3 * 448 * 640)


def avarage_loss_from_list3(list3):
    # list3 is trajectory losses per epoch
    # list2 is losses per trajectory
    # list1 is loss per frame of a trajectory
    # returns the epoch avarage for last frame loss.
    last_frame_loss = [[list1[-1] for list1 in list2] for list2 in list3]
    avarage_list1 = [sum(l) / len(l) for l in last_frame_loss]
    return avarage_list1


def sum_loss_from_list3(list3):
    # list3 is trajectory losses per epoch
    # list2 is losses per trajectory
    # list1 is loss per frame of a trajectory
    # returns the epoch sum for last frame loss.
    last_frame_loss = [[list1[-1] for list1 in list2] for list2 in list3]
    avarage_list1 = [sum(l) for l in last_frame_loss]
    return avarage_list1


def avg_loss_from_list2(list2):
    # list2 is losses per trajectory
    # list1 is loss per frame of a trajectory
    # returns the avarage for last frame loss.
    losses_per_frame = [list1[-1] for list1 in list2]
    # print(losses_per_frame)
    return sum(losses_per_frame) / len(losses_per_frame)


def cumul_sum_loss_from_list3(loss_list):
    return [sum([sum(trajectory_losses) for trajectory_losses in epoch_losses]) for epoch_losses in loss_list]


def factors_experiment():
    n_iter = 51
    flow_list = [i for i in np.linspace(start=1.0, stop=0.0, num=n_iter // 3)] + [0.0] * (n_iter - n_iter // 3)
    rot_list = [i for i in np.linspace(start=0.0, stop=1.0, num=n_iter // 3)] + [i for i in np.linspace(start=1.0,
                                                                                                        stop=0.0,
                                                                                                        num=n_iter // 3)] + [
                   0.0] * (n_iter - 2 * n_iter // 3)
    t_list = [0.0] * (n_iter - 2 * n_iter // 3) + [i for i in np.linspace(start=0.0, stop=1.0,
                                                                          num=n_iter // 3)] + [1.0] * (
                     n_iter - 2 * n_iter // 3)

    print(len(flow_list), len(rot_list), len(t_list))
    for i in range(n_iter):
        print(flow_list[i], rot_list[i], t_list[i])


def compare_results(loss_list_dir):
    def extract_label(filename):
        components = filename.split('_')
        return components[-1]

    for filename in os.listdir(loss_list_dir):
        print(filename)
        f = os.path.join(loss_list_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            loss_list = torch.load(f)
            loss_per_epoch = cumul_sum_loss_from_list3(loss_list)

            plt.plot(loss_per_epoch, label=extract_label(filename))
    plt.legend(loc='lower left', fontsize=6)
    print("showing")
    plt.show()


def image_experiment(pert_path):
    pert = cv2.cvtColor(cv2.imread(pert_path), cv2.COLOR_BGR2RGB)
    eps = 1000
    pert = torch.tensor(pert).to(torch.float)
    print(f"pertubation l2 norm: {torch.norm(pert)}")
    for i in range(10):
        pert = F.normalize(pert.view(pert.shape[0], -1), p=2, dim=-1).view(pert.shape) * eps
        print(f"pertubation l2 norm: {torch.norm(pert)}")


def see_flow_experiment():
    flow_path = '/home/nadav.nissim/final-project/code/results/kitti_custom/tartanvo_1914/VO_adv_project_train_dataset_8_frames/train_attack/universal_attack/gradient_ascent/attack_conv_norm_Linf/opt_whole_trajectory/opt_t_crit_none_factor_1_0_rot_crit_dot_product_factor_1_0_flow_crit_l1_factor_1_0_target_t_crit_none/eval_rms/eps_1_attack_iter_50_alpha_0_05/flow/1/00011_seed_00001_adv'
    flow = np.load(flow_path + '/000000.npy')
    pic = visflow(flow)
    plt.imshow(pic)
    plt.show()


def norm_experiment(pertubation_dir):
    for filename in os.listdir(pertubation_dir):
        eps = 100000
        f = os.path.join(pertubation_dir, filename)
        # checking if it is a file
        pert_list = []
        if os.path.isfile(f):
            pert = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
            pert = torch.tensor(pert).to(torch.float)
            pert_list.append(pert)
            norm = torch.norm(pert, p=2)

            diff = F.normalize(pert.view(pert.shape[0], -1), p=2, dim=-1).view(pert.shape) * norm / 21.16711 - pert

            # diff = F.normalize(pert, p=2) * norm / 36.66 - pert
            print(f'diff norm = {torch.norm(diff, p=2)}')
            print(
                f"pertubation l2 norm: {torch.norm(F.normalize(pert.view(pert.shape[0], -1), p=2, dim=-1).view(pert.shape), p=2)}")


def conv_pretrain():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h_dim = 3

    # model = PertGenerator(h_dim).to(device)
    # encoder = MyEncoder(h_dim).to(device)

    # encoder = MyEncoder_2(h_dim).to(device)
    # kernel_shape = encoder.output_shape

    # model = MyDecoder(h_dim, kernel_shape).to(device)
    # model = AdjDecoder(h_dim, kernel_shape).to(device)
    # model = LinConvGen()
    # model = FlowGen().to(device)
    model = PertGeneratorT().to(device)
    # print(f'encoder has {get_n_params(encoder)} parameters')
    print(f'model has {get_n_params(model)} parameters')

    pert_shape = [1, 3, 448, 640]
    num_epochs = 5000
    batch_size = 10
    lr = 0.001
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0000001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    targets = [(torch.randn(model.kernel.shape), torch.clamp(torch.randn(pert_shape), 0, 255).to(torch.float)) for i in
               range(batch_size)]
    # targets = {torch.clamp(torch.randn(pert_shape), 0, 255).to(torch.float) for i in range(5)}
    perts_per_round = []
    # for i in range(num_epochs):
    #     optimizer.zero_grad()
    #     for target in targets:
    #
    #         # pert = model.sample(device)
    #         # loss = loss_fn(pert, target.to(device))
    #         image =target.to(device)# torch.randn(pert_shape).to(torch.float).to(device)
    #         h = encoder(image)
    #         pert = model.sample(h)
    #         loss = loss_fn(pert, image)
    #         loss.backward()
    #
    #         # scheduler.step()
    #         print(f"epoch {i}, loss:{loss.item()}, lr: {optimizer.param_groups[0]['lr']}")
    #         if (i+1) % (num_epochs//4) == 0:
    #             perts_per_round.append(pert.detach().cpu())
    #     optimizer.step()
    print('---------')
    # for target in targets + targets:
    #     print(len(perts_per_round))
    #     for i in range(num_epochs):
    #         optimizer.zero_grad()
    #
    #         image =target.to(device)
    #         h = encoder(image)
    #         pert = model.sample(h)
    #         loss = loss_fn(pert, image)
    #         loss.backward()
    #         optimizer.step()
    #         # scheduler.step()
    #         if (i) % (num_epochs//10) == 0:
    #             print(f"epoch {i}, loss:{loss.item()}, lr: {optimizer.param_groups[0]['lr']}")
    #
    #     perts_per_round.append(model.sample(device).detach())

    # windows = [[130, 180], [50, 200,], [0,400],[0, 447]]

    for i in range(num_epochs):
        optimizer.zero_grad()
        for target in targets:
            h, image = target[0].to(device), target[1].to(device)
            # h = encoder(image)
            # print(f'h shape: {h.shape}')
            # pert = model.sample(device=device)
            pert = model(h)

            # print(f'pert shape: {pert.shape}')
            # window = [0,(1 + i//5)%488]
            # window = windows[1]#[i // 50]
            # print(f'window: {window}')
            # loss = loss_fn(pert[:,:,window,window], image[:,:,window,window])
            loss = loss_fn(pert, image)
            # print(f'loss: {loss.item()}')
            loss.backward()
        optimizer.step()
        # scheduler.step()
        if (i) % (num_epochs // 100) == 0:
            print(f"epoch {i}, loss:{loss.item()}, facror: {model.factor.item()}")

        if (i - 1) % (num_epochs // 5) == 0:
            # print(f"epoch {i}, loss:{loss.item()}, lr: {optimizer.param_groups[0]['lr']}")
            perts_per_round.append(model.sample(device).detach())

    for i, p in enumerate(perts_per_round):
        save_image(p, 'results/perts_per_round/pert_per_round_' + str(i) + '.png')
    save_image(targets[0][1], 'results/perts_per_round/target.png')
    torch.save(model, 'results/pert_generator.pt')
    # torch.save(encoder, 'results/encoder.pt')


def load_gen_and_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('results/pert_generator.pt', map_location=device)
    for p in model.parameters():
        p.requires_grad = False
    model.kernel.requires_grad = True
    model.factor.requires_grad = True
    print([p.requires_grad for p in model.parameters()])
    pert_shape = [1, 3, 448, 640]
    lr = 0.001
    num_epochs = 5000
    optimizer = torch.optim.Adam(model.parameters(), lr)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.L1Loss()
    singular_target = torch.clamp(torch.randn(pert_shape), 0, 255).to(torch.float).to(device)
    for i in range(num_epochs):
        optimizer.zero_grad()
        pert = model.sample(device)
        loss = loss_fn(pert, singular_target)
        loss.backward()
        optimizer.step()
        if (i) % (num_epochs // 100) == 0:
            print(f"epoch {i}, loss:{loss.item()}, factor: {model.factor.item()}")
    save_image(model.sample(device), 'results/perts_per_round/final' + '.png')
    save_image(singular_target[0], 'results/perts_per_round/single_target.png')


def imshow(tensor_img):
    plt.imshow(tensor_img.detach().permute(1, 2, 0))
    plt.show()


def load_conv_experiment():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('results/pert_generator.pt', map_location=device)

    output = model.sample(device, 200 * torch.randn([1, 3, 2, 2]))[0]
    plt.imshow(output.detach().permute(1, 2, 0))
    plt.show()
    output = model.sample(device, torch.randn(model.kernel.shape))[0]
    plt.imshow(output.detach().permute(1, 2, 0))
    plt.show()


def optimize_flow_experiment():
    from loss import VOCriterion
    from Network.VOFlowNet import VOFlowRes as FlowPoseNet
    from Network.PWC import PWCDCNet as FlowNet
    torch.manual_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_shape = [1, 3, 448, 640]
    flow_shape = [1, 2, 112, 160]
    flowPoseNet = FlowPoseNet().to(device)
    intrinsic_I0 = torch.load('intrinsic_I0.pt').to(device)
    intrinsic = intrinsic_I0[0].unsqueeze(0)
    # flow = flowNet.to(device)(torch.randn(input_shape).to(device), torch.randn(input_shape).to(device))
    # print(f'flow shape: {flow_shape}')
    flow = torch.nn.Parameter(torch.randn(flow_shape, requires_grad=True).to(device))
    flow_input = torch.cat((flow, intrinsic), dim=1)
    output = flowPoseNet(flow_input)
    # print(output)

    num_epochs = 100
    lr = 1.001
    lr2 = 0.001
    optimizer = torch.optim.Adam([flow], lr)
    loss_fn = VOCriterion()
    default_motions = torch.zeros_like(output).to(device)
    target_pose = torch.tensor([[5.0000e+01, 7.8015e-06, -1.6291e-06]]).to(device)
    scale = torch.tensor([0.1667]).to(device)
    first_up_loss = None
    last_up_loss = None
    first_down_loss = None
    last_down_loss = None
    for i in range(num_epochs):
        optimizer.zero_grad()
        flow_input = torch.cat((flow, intrinsic), dim=1)
        output = flowPoseNet(flow_input)
        loss = -loss_fn((output, flow), scale, default_motions, target_pose)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        if first_up_loss is None:
            first_up_loss = -loss.item()
        last_up_loss = -loss.item()
        if (i) % (num_epochs // 10) == 0:
            print(f"epoch {i}, loss:{-loss.item()}")

    flow.requires_grad = False
    # target_flow = torch.randn(flow_shape).to(device)
    flownet = FlowNet().to(device)
    img1 = torch.clamp(torch.randn(img_shape).to(device),0,1)
    img2 = torch.clamp(torch.randn(img_shape).to(device),0,1)
    patch = torch.nn.Parameter(torch.clamp(torch.randn(1, 3, 100, 100, requires_grad=True),0,1).to(device))
    optimizer2 = torch.optim.Adam([patch], lr2)
    # loss_fn2 = nn.MSELoss()
    loss_fn2 = nn.L1Loss()


    for i in range(10*num_epochs):
        optimizer2.zero_grad()
        img1[:, :, 200:300, 200:300] = patch
        img2[:, :, 200:300, 200:300] = patch
        # img1 = torch.clamp(img1, 0, 1)
        # img2 = torch.clamp(img2, 0, 1)
        output = flownet(img1, img2)
        loss = loss_fn2(output, flow)
        loss.backward(retain_graph=True)
        optimizer2.step()
        # patch.data = torch.clamp(patch.data, 0, 1)
        if first_down_loss is None:
            first_down_loss = loss.item()
        last_down_loss = loss.item()
        if (i) % (num_epochs // 10) == 0:
            print(f"epoch {i}, loss:{loss.item()}")


    img1 = torch.randn(img_shape).to(device)
    img2 = torch.randn(img_shape).to(device)
    output_f = flownet(img1, img2)
    flow_input = torch.cat((output_f, intrinsic), dim=1)
    output = flowPoseNet(flow_input)
    # print(f'clean output: {output}')
    img1[:, :, 200:300, 200:300] = patch
    img2[:, :, 200:300, 200:300] = patch
    output_f = flownet(img1, img2)
    flow_input = torch.cat((output_f, intrinsic), dim=1)
    output = flowPoseNet(flow_input)
    # print(f'perturbed output: {output}')

    print(f'up loss diff: {last_up_loss - first_up_loss}')
    print(f'down loss diff: {last_down_loss - first_down_loss}')


def parameters_experiment():
    gen = PertGenerator()
    print([p.requires_grad for p in gen.parameters()])
    output = gen.sample(torch.device('cpu'))
    print(output.shape)
    torchvision.utils.save_image(output, 'results/sandbox/initial_pert.png')


def main():
    optimize_flow_experiment()
    return
    # factors_experiment()
    # see_flow_experiment()
    # experiment_tartan()
    # experiment_conv_sizes()

    # plot_experiment()
    # image_experiment('pertubations/pgd-500iter.png')
    # norm_experiment('pertubations')
    # random_initiation()
    # compare_results('results/loss_lists')

    #
    # reversed_experiment()

    conv_pretrain()
    load_gen_and_train()
    return

    # load_conv_experiment()
    # random_initiation()

    class ImgGen(nn.Module):
        def __init__(self, shape=(3, 64, 64), seed_shape=(3, 32, 32)):
            super(ImgGen, self).__init__()
            self.seed = torch.randn(*seed_shape)
            self.output_shape = shape
            self.output_num = torch.ones(self.output_shape).flatten().shape[0]
            self.linear = torch.nn.Linear(self.seed.flatten().shape[0], self.output_num)
            self.linear.weight.data.uniform_(-0.01, 0.03)

        def sample(self):
            return self.forward(self.seed)

        def forward(self, x):
            if len(x.shape) == 1:
                return self.linear(x).view(self.output_shape)
            else:
                output = self.linear(x.flatten())
                return output.view(self.output_shape)

    class SeqGen(nn.Module):
        def __init__(self, ):
            super(SeqGen, self).__init__()
            self.seed = torch.randn(3, 2, 2)
            linear1 = ImgGen(shape=(3, 4, 4), seed_shape=(3, 2, 2))
            linear2 = ImgGen(shape=(3, 8, 8), seed_shape=(3, 4, 4))
            linear3 = ImgGen(shape=(3, 16, 16), seed_shape=(3, 8, 8))
            linear4 = ImgGen(shape=(3, 32, 32), seed_shape=(3, 16, 16))
            linear5 = ImgGen(shape=(3, 64, 64), seed_shape=(3, 32, 32))
            # self.convt1 = nn.ConvTranspose2d(3, 3, kernel_size=(387, 579), stride=1, padding=1)
            conv1 = nn.ConvTranspose2d(3, 3, kernel_size=(10, 10), stride=1, padding=0)
            conv2 = nn.ConvTranspose2d(3, 3, kernel_size=(10, 10), stride=1, padding=0)
            conv3 = nn.ConvTranspose2d(3, 3, kernel_size=(10, 10), stride=1, padding=0)

            self.output_shape = (3, 64, 64)
            self.sequential = torch.nn.Sequential(linear1, linear2, linear3, linear4, linear5)

        def sample(self):
            return self.forward(self.seed)

        def forward(self, x):
            if len(x.shape) == 1:
                x = self.sequential(x).view(self.output_shape)
            else:
                output = self.sequential(x.flatten())
                x = output.view(self.output_shape)
            x = x.unsqueeze(0)
            return self.convt1(x)[0]

    class MLP_GEN(nn.Module):
        def __init__(self, ):
            super(MLP_GEN, self).__init__()
            self.seed = torch.FloatTensor(3, 2, 2).uniform_(0, 1)
            linear1 = ImgGen(shape=(3, 16, 16), seed_shape=(3, 2, 2))
            linear2 = ImgGen(shape=(3, 448, 640), seed_shape=(3, 16, 16))
            self.output_shape = (3, 448, 640)
            self.sequential = torch.nn.Sequential(linear1, linear2)

        def sample(self):
            return self.forward(self.seed)

        def forward(self, x):
            if len(x.shape) == 1:
                x = self.sequential(x).view(self.output_shape)
            else:
                output = self.sequential(x.flatten())
                x = output.view(self.output_shape)
            return x

    def linear(in_planes, out_planes):
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.ReLU(inplace=True)
        )

    # imshow(linear1.sample())
    # print(linear1.sample().max())
    # print(linear1.sample().min())
    # normalize_generator(linear1)
    # linear1 =normalize_generator(linear1)
    # imshow(linear1.sample())
    # print(linear1.sample().max())
    # print(linear1.sample().min())

    # seq = MLP_GEN()
    seq = LinConvGen()
    print(f'seq has {get_n_params(seq)} params')
    output = seq.sample()
    print(output.shape)
    imshow(output)
    print(output.max())
    print(output.min())
    normalize_generator(seq)
    output = seq.sample()
    imshow(output)
    print(output.max())
    print(output.min())
    print("done")


def normalize_generator(generator):
    def myloss(tensor_image):
        zeros = torch.zeros(tensor_image.shape)
        upper_error = torch.nn.L1Loss()(torch.nn.ReLU()(tensor_image - 0.9), zeros)
        lower_error = torch.nn.L1Loss()(torch.nn.ReLU()(0.1 + (-1) * tensor_image), zeros)
        print(lower_error.item(), upper_error.item())
        return lower_error + upper_error

    optimizer = torch.optim.SGD(generator.parameters(), lr=0.1, momentum=0.9)
    for i in range(14):
        output = generator.sample()
        # output = generator(torch.randn(generator.seed.shape))
        loss = myloss(output)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch {i}, loss:{loss.item()}")
    return generator


if __name__ == '__main__':
    main()
