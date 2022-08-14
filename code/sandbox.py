# this file is for experiments like checking tensor sizes and stuff
import os

import cv2
import matplotlib.pyplot as plt
from torch import nn
from torchvision.utils import save_image

sandbox_backend = plt.get_backend()
import numpy as np
import torch
from torch.nn import ConvTranspose2d, Sequential
from torch.nn import functional as F

from Datasets.utils import visflow


plt.switch_backend(sandbox_backend)


class MyEncoder(torch.nn.Module):
    def __init__(self,out_channels):
        super(MyEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, (70, 100), 8, (1, 4))
        self.conv2 = nn.Conv2d(10, 30, 4, 1, 1)
        self.conv3 = nn.Conv2d(30, out_channels, (35, 50), 3, (4, 5))


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class MyEncoder_2(torch.nn.Module):
    def __init__(self, out_channels):
        super(MyEncoder_2, self).__init__()
        channels = [3,1,1,1,1,out_channels]
        self.conv1 = nn.Conv2d(channels[0], channels[1], (70, 100), 1, 0)
        self.conv2 = nn.Conv2d(channels[1], channels[2], (70, 100), 1, 0)
        self.conv3 = nn.Conv2d(channels[2], channels[3], (70, 100), 1, 0)
        self.conv4 = nn.Conv2d(channels[3], channels[4], (70, 100), 1, 0)
        self.conv5 = nn.Conv2d(channels[4], channels[5], (70, 100), 1, 0)

        temp = torch.randn(1, 3, 448, 640)
        self.output_shape = self.forward(temp).shape



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class MyDecoder(torch.nn.Module):
    def __init__(self,in_channels, kernel_shape):
        super(MyDecoder, self).__init__()
        channels = [in_channels, 1, 1, 1, 1, 3]

        # self.conv1 = nn.ConvTranspose2d(channels[0], channels[1], (70, 100), 1, 0)
        # self.conv2 = nn.ConvTranspose2d(channels[1], channels[2], (70, 100), 1, 0)
        # self.conv3 = nn.ConvTranspose2d(channels[2], channels[3], (70, 100), 1, 0)
        # self.conv4 = nn.ConvTranspose2d(channels[3], channels[4], (70, 100), 1, 0)
        # self.conv5 = nn.ConvTranspose2d(channels[4], channels[5], (70, 100), 1, 0)
        self.conv1 = ConvTranspose2d(in_channels, 20, kernel_size=(1, 1), stride=(1, 1), padding=(0, 1))
        self.fc1 = nn.Linear(20, kernel_shape[0])
        self.conv2 = ConvTranspose2d(20, 30, kernel_size=(14, 25), stride=(2, 2), padding=(1, 4))
        self.conv3 = ConvTranspose2d(30, 20, kernel_size=(35, 50), stride=(3, 3), padding=(4, 7))
        self.conv4 = ConvTranspose2d(20, 3, kernel_size=(70, 100), stride=(4, 4), padding=(1, 4))



        self.kernel_shape = kernel_shape

    def sample(self, device, seed=None):
        if seed is None:
            pert = self.forward(torch.ones(self.kernel_shape).to(device))
        else:
            pert = self.forward(seed)
        return torch.clamp(pert, 0, 255)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        return x

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
        # c1 = ConvTranspose2d(1, 20, kernel_size=(1, 1), stride=(1, 1), padding=(0, 1))
        c2 = ConvTranspose2d(h_dim, 30, kernel_size=(14, 25), stride=(2, 2), padding=(1, 4))
        c3 = ConvTranspose2d(30, 20, kernel_size=(35, 50), stride=(3, 3), padding=(4, 7))
        
        c4 = ConvTranspose2d(20, 3, kernel_size=(70, 100), stride=(4, 4), padding=(1, 4))
        self.cnn = Sequential(c2, c3, c4)

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
            pert = self.cnn(self.kernel.to(device))
        else:
            pert = self.cnn(seed)
        return torch.clamp(pert, 0, 255)


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
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True)
        )


class ReversedBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(ReversedBasicBlock, self).__init__()

        self.conv1 = reversed_conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.ConvTranspose2d(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return F.relu(out, inplace=True)

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

    def linear(in_planes, out_planes):
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
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
                downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                                       kernel_size=1, stride=stride)

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

            return nn.Sequential(*layers)

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


    dlayers = 5*[nn.ConvTranspose2d(1, 1, (70, 100), 1, 0)]
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
        up_layers.append(ConvTranspose2d(channels[i], channels[i+1], kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1))
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


def plot_experiment():
    adv_loss_list = [[[0.0, 0.14862024784088135, 0.18555615842342377, 0.17333632707595825, 0.24865370988845825,
                       0.23890715837478638, 0.39017316699028015, 0.41088372468948364],
                      [0.0, 0.22953064739704132, 0.22048819065093994, 0.26680365204811096, 0.2919774055480957,
                       0.2518039345741272, 0.2735888659954071, 0.28738802671432495],
                      [0.0, 0.053253769874572754, 0.0923033133149147, 0.10666584968566895, 0.17330749332904816,
                       0.21534611284732819, 0.30499333143234253, 0.35653337836265564],
                      [0.0, 0.06741935759782791, 0.04401342198252678, 0.059521812945604324, 0.08702347427606583,
                       0.19664646685123444, 0.2714165449142456, 0.31367039680480957],
                      [0.0, 0.18394583463668823, 0.1619003415107727, 0.30507442355155945, 0.33068758249282837,
                       0.37187179923057556, 0.43126368522644043, 0.4660124182701111],
                      [0.0, 0.09174749255180359, 0.23084312677383423, 0.23142243921756744, 0.34334197640419006,
                       0.3853986859321594, 0.49055230617523193, 0.6568774580955505],
                      [0.0, 0.054058369249105453, 0.07944726198911667, 0.1245184987783432, 0.13291317224502563,
                       0.27071917057037354, 0.2872328758239746, 0.3017038106918335],
                      [0.0, 0.020408552139997482, 0.050799015909433365, 0.11006343364715576, 0.17893019318580627,
                       0.1636805534362793, 0.1640065461397171, 0.23063990473747253],
                      [0.0, 0.08164269477128983, 0.2453651875257492, 0.2522113621234894, 0.23335014283657074,
                       0.4307975471019745, 0.6437662839889526, 0.8125373721122742],
                      [0.0, 0.13440962135791779, 0.16999270021915436, 0.27784299850463867, 0.3733164370059967,
                       0.5401980876922607, 0.6617853045463562, 0.8084887862205505]], [
                         [0.0, 0.0643763616681099, 0.14176705479621887, 0.20819222927093506, 0.20527194440364838,
                          0.22366297245025635, 0.23762653768062592, 0.3606208264827728],
                         [0.0, 0.2223089635372162, 0.2566027343273163, 0.35270461440086365, 0.4015583395957947,
                          0.36061540246009827, 0.37777894735336304, 0.34292715787887573],
                         [0.0, 0.12496976554393768, 0.10953831672668457, 0.17529161274433136, 0.14636650681495667,
                          0.3555598556995392, 0.3591381907463074, 0.34962478280067444],
                         [0.0, 0.1661892831325531, 0.13084784150123596, 0.12474742531776428, 0.1948155164718628,
                          0.183126762509346, 0.2458329051733017, 0.2651987671852112],
                         [0.0, 0.21152296662330627, 0.1400085687637329, 0.26528477668762207, 0.3525626063346863,
                          0.31600120663642883, 0.39996758103370667, 0.4647858142852783],
                         [0.0, 0.18673202395439148, 0.3346666991710663, 0.33231493830680847, 0.44484907388687134,
                          0.5316987037658691, 0.5669482946395874, 0.7012112736701965],
                         [0.0, 0.060651928186416626, 0.0951203778386116, 0.1305135190486908, 0.1399933248758316,
                          0.24551880359649658, 0.28150245547294617, 0.30435073375701904],
                         [0.0, 0.038493216037750244, 0.07319360226392746, 0.19484643638134003, 0.2447516769170761,
                          0.2454325258731842, 0.43177172541618347, 0.5145484209060669],
                         [0.0, 0.11112399399280548, 0.30523252487182617, 0.2526319921016693, 0.285179078578949,
                          0.40011417865753174, 0.6642181873321533, 0.8205310106277466],
                         [0.0, 0.09405559301376343, 0.18501213192939758, 0.2592082619667053, 0.28949347138404846,
                          0.46547064185142517, 0.6699560880661011, 0.6978067755699158]], [
                         [0.0, 0.058610379695892334, 0.13775582611560822, 0.19455742835998535, 0.19487088918685913,
                          0.20978356897830963, 0.22847282886505127, 0.35277044773101807],
                         [0.0, 0.21083301305770874, 0.25496867299079895, 0.3755926787853241, 0.4290325343608856,
                          0.3774901330471039, 0.41648393869400024, 0.37232887744903564],
                         [0.0, 0.12080956995487213, 0.10981106758117676, 0.17905977368354797, 0.14900875091552734,
                          0.38111412525177, 0.39123809337615967, 0.36554670333862305],
                         [0.0, 0.172256201505661, 0.13760408759117126, 0.13733917474746704, 0.2414688766002655,
                          0.2083907276391983, 0.2636767327785492, 0.28999602794647217],
                         [0.0, 0.2264130413532257, 0.15971730649471283, 0.28945279121398926, 0.38339856266975403,
                          0.3508726954460144, 0.4296099543571472, 0.5172503590583801],
                         [0.0, 0.20139965415000916, 0.37513160705566406, 0.38104408979415894, 0.5091618299484253,
                          0.5986361503601074, 0.6104772686958313, 0.7209340333938599],
                         [0.0, 0.050770021975040436, 0.10856734961271286, 0.1515529602766037, 0.1439608633518219,
                          0.26329362392425537, 0.2987309694290161, 0.2977648377418518],
                         [0.0, 0.03998294472694397, 0.07476917654275894, 0.20487160980701447, 0.2704195976257324,
                          0.287880539894104, 0.4572337865829468, 0.5514441132545471],
                         [0.0, 0.10191790014505386, 0.3123597800731659, 0.28423529863357544, 0.3441764712333679,
                          0.48709404468536377, 0.7454376816749573, 0.9219976663589478],
                         [0.0, 0.1223582774400711, 0.22582657635211945, 0.32687684893608093, 0.3862111270427704,
                          0.5713242292404175, 0.7894744873046875, 0.8335668444633484]], [
                         [0.0, 0.06041986867785454, 0.1355874240398407, 0.206338569521904, 0.21481014788150787,
                          0.22621794044971466, 0.25252944231033325, 0.3897670805454254],
                         [0.0, 0.21869876980781555, 0.2616017162799835, 0.35982343554496765, 0.41021454334259033,
                          0.3666437268257141, 0.39095592498779297, 0.35715511441230774],
                         [0.0, 0.1386849582195282, 0.12533871829509735, 0.21549756824970245, 0.1720481961965561,
                          0.4601077437400818, 0.4754260182380676, 0.45001548528671265],
                         [0.0, 0.16987285017967224, 0.13130582869052887, 0.12649770081043243, 0.1874154955148697,
                          0.18245357275009155, 0.25434213876724243, 0.28254690766334534],
                         [0.0, 0.24218738079071045, 0.18979234993457794, 0.3223373591899872, 0.47339683771133423,
                          0.43979325890541077, 0.5151149034500122, 0.6399728059768677],
                         [0.0, 0.19979235529899597, 0.35850343108177185, 0.36903294920921326, 0.5052971839904785,
                          0.6024996042251587, 0.6264953017234802, 0.7391652464866638],
                         [0.0, 0.03646143525838852, 0.09535840153694153, 0.1365174651145935, 0.15709288418293,
                          0.27068597078323364, 0.31824791431427, 0.3273587226867676],
                         [0.0, 0.04243713617324829, 0.07463100552558899, 0.18900492787361145, 0.2549803853034973,
                          0.27626127004623413, 0.4504165053367615, 0.5214200615882874],
                         [0.0, 0.09349887818098068, 0.30928125977516174, 0.28682786226272583, 0.3353237509727478,
                          0.48066699504852295, 0.739190399646759, 0.9182389974594116],
                         [0.0, 0.09392382204532623, 0.2000541388988495, 0.30498233437538147, 0.34046801924705505,
                          0.5486984848976135, 0.7827267050743103, 0.8401306867599487]]]
    rms_target_list = avarage_loss_from_list3(adv_loss_list)
    plt.plot(rms_target_list)
    plt.show()


def cumul_sum_loss_from_list3(loss_list):
    return [sum([sum(trajectory_losses) for trajectory_losses in epoch_losses]) for epoch_losses in loss_list]


def factors_experiment():
    n_iter = 51
    flow_list = [i for i in np.linspace(start=1.0, stop=0.0, num=n_iter // 3)] + [0.0] * (n_iter - n_iter // 3)
    rot_list = [i for i in np.linspace(start=0.0, stop=1.0, num=n_iter // 3)] + [i for i in np.linspace(start=1.0,
                                                                                                          stop=0.0,
                                                                                                          num=n_iter // 3)] + [0.0] * (n_iter - 2 * n_iter // 3)
    t_list = [0.0] * (n_iter - 2*n_iter // 3) + [i for i in np.linspace(start=0.0, stop=1.0,
                                                                      num=n_iter // 3)] + [1.0] * (
                     n_iter - 2 * n_iter // 3)


    print(len(flow_list), len(rot_list), len(t_list))
    for i in range(n_iter):
        print(flow_list[i], rot_list[i], t_list[i])

def compare_experiment(loss_list_dir):
    for filename in os.listdir(loss_list_dir):
        print(filename)
        f = os.path.join(loss_list_dir, filename)
        # checking if it is a file
        if os.path.isfile(f):
            loss_list = torch.load(f)
            loss_per_epoch = cumul_sum_loss_from_list3(loss_list)
            plt.plot(loss_per_epoch, label=filename)
    plt.legend(loc='lower right', fontsize=6)
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
            print(f'diff norm = {torch.norm(diff,p=2)}')
            print(f"pertubation l2 norm: {torch.norm( F.normalize(pert.view(pert.shape[0], -1), p=2, dim=-1).view(pert.shape), p=2)}")

def conv_pretrain():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h_dim = 3
    model = PertGenerator(h_dim).to(device)
    encoder = MyEncoder(h_dim).to(device)
    # encoder = MyEncoder_2(h_dim).to(device)
    # kernel_shape = encoder.output_shape
    # print(kernel_shape)
    # model = MyDecoder(h_dim, kernel_shape).to(device)

    # print(f'encoder has {get_n_params(encoder)} parameters')
    print(f'model has {get_n_params(model)} parameters')

    pert_shape = [1, 3, 448, 640]
    num_epochs = 1000
    batch_size = 1
    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.00005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0000001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.MSELoss()
    targets = [torch.clamp(torch.randn(pert_shape), 0, 255).to(torch.float) for i in range(batch_size)]
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

    for i in range(num_epochs):
        optimizer.zero_grad()
        for target in targets:
            image = target.to(device)
            h = encoder(image)
            pert = model.sample(device)        # TODO: notice H is missing
            loss = loss_fn(pert, image)

            loss.backward()
        optimizer.step()
        # scheduler.step()
        if (i) % (num_epochs // 100) == 0:
            print(f"epoch {i}, loss:{loss.item()}, lr: {optimizer.param_groups[0]['lr']}")

        if (i - 1) % (num_epochs // 5) == 0:
            # print(f"epoch {i}, loss:{loss.item()}, lr: {optimizer.param_groups[0]['lr']}")
            perts_per_round.append(model.sample(device).detach())


    for i, p in enumerate(perts_per_round):
        save_image(p, 'results/perts_per_round/pert_per_round_'+str(i)+'.png')
    torch.save(model, 'results/pert_generator.pt')
    torch.save(encoder, 'results/encoder.pt')

def load_conv_experiment():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('results/pert_generator.pt', map_location=device)

    output = model.sample(device)[0]
    plt.imshow(output.detach().permute(1, 2, 0))
    plt.show()
    output = model.sample(device, torch.randn(model.kernel.shape))[0]
    plt.imshow(output.detach().permute(1, 2, 0))
    plt.show()


def main():
    # factors_experiment()
    # see_flow_experiment()
    # experiment_tartan()
    # experiment_conv_sizes()

    # plot_experiment()
    # image_experiment('pertubations/pgd-500iter.png')
    # norm_experiment('pertubations')
    compare_experiment('results/loss_lists')
    #
    # conv_pretrain()
    # load_conv_experiment()
    print("done")


if __name__ == '__main__':
    main()
