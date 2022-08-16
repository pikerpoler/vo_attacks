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
        return torch.clamp(x, min=0, max=2)




def random_initiation():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = AdjEncoder(out_channels=3).to(device)
    kernel_shape = encoder.output_shape
    print(f'kernel shape: {kernel_shape}')
    model = AdjDecoder(in_channels=3, kernel_shape=kernel_shape).to(device)
    return model

kernel_sizes = 10 * [(3,3)]
inner_channels = [200, 200, 150, 150, 100, 100, 50, 50, 25, 25, 3, 3]
strides = 5 * [1, 2]

class AdjEncoder(torch.nn.Module):
    #this class' whole purpose is to calculate the kernel shape
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

        return torch.clamp(self.cnn(x), 0, 255)

    def sample(self, device, seed=None):
        if seed is None:
            pert = self.forward(self.kernel_value * torch.ones(self.kernel_shape).to(device))
        else:
            pert = self.forward(seed)
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
        self.fc = nn.Linear(output_size, output_size)   # c1 = ConvTranspose2d(1, 20, kernel_size=(1, 1), stride=(1, 1), padding=(0, 1))
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
    plt.legend(loc='upper left', fontsize=5)
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
    lr = 0.001
    batch_size = 1
    # model = PertGenerator(h_dim).to(device)
    # encoder = MyEncoder(h_dim).to(device)

    # encoder = MyEncoder_2(h_dim).to(device)
    # kernel_shape = encoder.output_shape

    encoder = AdjEncoder(h_dim).to(device)
    kernel_shape = encoder.output_shape
    print(f'kernel_shape = {kernel_shape}')
    # model = MyDecoder(h_dim, kernel_shape).to(device)
    model = AdjDecoder(h_dim, kernel_shape).to(device)

    # print(f'encoder has {get_n_params(encoder)} parameters')
    print(f'model has {get_n_params(model)} parameters')

    pert_shape = [1, 3, 448, 640]
    num_epochs = 100
    # batch_size = 3
    params = list(model.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
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
            # print(f'h shape: {h.shape}')
            pert = model.sample(device=device, seed=h)
            # print(f'pert shape: {pert.shape}')
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

    output = model.sample(device, 200*torch.randn([1, 3, 2, 2]))[0]
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
    random_initiation()
    compare_results('results/loss_lists')
    #
    # conv_pretrain()
    # load_conv_experiment()
    # random_initiation()
    print("done")


if __name__ == '__main__':
    main()
