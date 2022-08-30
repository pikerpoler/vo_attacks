import os

import numpy as np
import torch
from Datasets.tartanTrajFlowDataset import extract_traj_data
from attacks.attack import Attack
import time

from torch import nn
from torch.nn import ConvTranspose2d, Sequential
from torchvision.utils import save_image
from tqdm import tqdm
import cv2


def normalize_generator(generator):
    def myloss(tensor_image):
        zeros = torch.zeros(tensor_image.shape)
        upper_error = torch.nn.L1Loss()(torch.nn.ReLU()(tensor_image - 0.9), zeros)
        lower_error = torch.nn.L1Loss()(torch.nn.ReLU()(0.1 + (-1) * tensor_image), zeros)
        print(lower_error.item(), upper_error.item())
        return lower_error + upper_error

    optimizer = torch.optim.SGD(generator.parameters(), lr=0.01, momentum=0.9)
    for i in range(30):
        output = generator.sample()
        # output = generator(torch.randn(generator.seed.shape))
        loss = myloss(output)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch {i}, loss:{loss.item()}")
    return generator

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

class MLP_GEN(nn.Module):
    def __init__(self):
        super(MLP_GEN, self).__init__()
        seed_dim = 2
        h_dim = 8
        self.seed = torch.FloatTensor(3, seed_dim, seed_dim).uniform_(0, 1)
        linear1 = ImgGen(shape=(3, h_dim, h_dim), seed_shape=(3, seed_dim, seed_dim))
        relu = nn.ReLU()
        linear2 = ImgGen(shape=(3, 448, 640), seed_shape=(3, h_dim, h_dim))
        self.output_shape = (3, 448, 640)
        self.sequential = torch.nn.Sequential(linear1,relu, linear2)

    def sample(self, device):
        return self.forward(self.seed.to(device))

    def forward(self, x):
        if len(x.shape) == 1:
            x = self.sequential(x).view(self.output_shape)
        else:
            output = self.sequential(x.flatten())
            x = output.view(self.output_shape)
        return x

class PertGenerator(torch.nn.Module):

    '''
    this code is not clean, nor modular.
    '''
    def __init__(self, pert_shape):
        super(PertGenerator, self).__init__()
        self.pert_shape = pert_shape

        '''
        old architecture version:
        self.kernel = torch.ones(1, 1, 1, 1)
        c1 = ConvTranspose2d(1, 1, kernel_size=(70, 100), stride=(1, 1), padding=0)
        c2 = ConvTranspose2d(1, 2, kernel_size=(70, 100), stride=(1, 1), padding=0)
        c3 = ConvTranspose2d(2, 3, kernel_size=(35, 47), stride=(3, 3), padding=1, output_padding=1)

        self.cnn = Sequential(c1, c2, c3)
        '''

        '''
        # best so far:
        # Average target_rms_crit_adv_delta = 0.11246040026657284
        # Average mean_partial_rms_crit_adv_delta = 0.4526335008442402
        
        c1 = ConvTranspose2d(1, 20, kernel_size=(7, 10), stride=(1, 1), padding=(0, 1))
        c2 = ConvTranspose2d(20, 30, kernel_size=(14, 25), stride=(2, 2), padding=(1, 2))
        c3 = ConvTranspose2d(30, 20, kernel_size=(35, 50), stride=(3, 3), padding=(4, 7))
        c4 = ConvTranspose2d(20, 3, kernel_size=(70, 100), stride=(4, 4), padding=(1, 4))
        self.cnn = Sequential(c1, c2, c3, c4)

        c1 = ConvTranspose2d(1, 3, kernel_size=(448, 640), stride=(1, 1), padding=(0, 0))
        self.cnn = Sequential(c1)
        
        '''

        self.kernel = torch.nn.Parameter(100 * torch.ones(1, 200, 14, 20))
        channels = [200, 150, 100, 50, 25, 3]
        up_layers = []
        for i in range(5):
            up_layers.append(ConvTranspose2d(channels[i], channels[i], kernel_size=3, stride=1, padding=1, dilation=1))
            # up_layers.append(torch.nn.ReLU())  # without its 102 and with its 103
            up_layers.append(torch.nn.Dropout(0.1))
            up_layers.append(
                ConvTranspose2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, dilation=1,
                                output_padding=1))
        self.cnn = Sequential(*up_layers)

    def sample(self, device):
        return torch.clamp(self.cnn(self.kernel.to(device)),0 , 1)


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
        return torch.clamp(self.cnn(x), min=0, max=1)

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

        return torch.clamp(self.cnn(x), 0, 1)

    def sample(self, device, seed=None):
        if seed is None:
            pert = self.forward(self.kernel_value * torch.ones(self.kernel_shape).to(device))
        else:
            pert = self.forward(seed)
        return torch.clamp(pert, 0, 1)


class ConvPGD(Attack):
    def __init__(
            self,
            model,
            criterion,
            test_criterion,
            data_shape,
            norm='Linf',
            n_iter=20,
            n_restarts=1,
            alpha=None,
            rand_init=False,
            sample_window_size=None,
            sample_window_stride=None,
            pert_padding=(0, 0),
            init_pert_path=None,
            init_pert_transform=None,
            run_name='',
            generator=None,

    ):
        super(ConvPGD, self).__init__(model, criterion, test_criterion, norm, data_shape,
                                  sample_window_size, sample_window_stride,
                                  pert_padding, run_name)

        self.alpha = alpha

        self.n_restarts = n_restarts
        self.n_iter = n_iter
        if generator is not None:
            self.generator = generator
        else:
            self.generator = PertGenerator(data_shape)
        # self.generator = random_initiation()
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=alpha)

        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, n_iter, 1e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, gamma=2)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: alpha/np.log(epoch + 2))

    def gradient_ascent_step(self, data_shape, data_loader, y_list, clean_flow_list,eps ,device=None):

        self.optimizer.zero_grad()
        for data_idx, data in enumerate(data_loader):
            dataset_idx, dataset_name, traj_name, traj_len, \
            img1_I0, img2_I0, intrinsic_I0, \
            img1_I1, img2_I1, intrinsic_I1, \
            img1_delta, img2_delta, \
            motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
            mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)

            pert = self.generator.sample(device)
            pert = self.project(pert, eps)
            pert_expand = pert.expand(data_shape[0], -1, -1, -1).to(device)


            img1_adv, img2_adv, output_adv = self.perturb_model_single(pert_expand, img1_I0, img2_I0,
                                                                       intrinsic_I0,
                                                                       img1_delta, img2_delta,
                                                                       scale,
                                                                       mask1, mask2,
                                                                       perspective1,
                                                                       perspective2,
                                                                       device)
            loss = self.criterion(output_adv, scale.to(device), y_list[data_idx].to(device), patch_pose.to(device),
                                  clean_flow_list[data_idx].to(device))
            loss_sum = -loss.sum(dim=0)
            loss_sum.backward()
            # self.generator = normalize_generator(self.generator)

            del img1_I0
            del img2_I0
            del intrinsic_I0
            del img1_I1
            del img2_I1
            del intrinsic_I1
            del img1_delta
            del img2_delta
            del motions_gt
            del scale
            del pose_quat_gt
            del patch_pose
            del mask
            del perspective
            del img1_adv
            del img2_adv
            del output_adv
            torch.cuda.empty_cache()

        self.optimizer.step()
        # self.scheduler.step()



    def perturb(self, data_loader, y_list, eps,
                                   targeted=False, device=None, eval_data_loader=None, eval_y_list=None):

        a_abs = np.abs(eps / self.n_iter) if self.alpha is None else np.abs(self.alpha)
        multiplier = -1 if targeted else 1
        print("computing PGD attack with parameters:")
        print("attack random restarts: " + str(self.n_restarts))
        print("attack epochs: " + str(self.n_iter))
        print("attack norm: " + str(self.norm))
        print("attack epsilon norm limitation: " + str(eps))
        print("attack step size: " + str(a_abs))

        data_shape, dtype, eval_data_loader, eval_y_list, clean_flow_list, \
        eval_clean_loss_list, traj_clean_loss_mean_list, clean_loss_sum, \
        best_pert, best_loss_list, best_loss_sum, all_loss, all_best_loss = \
            self.compute_clean_baseline(data_loader, y_list, eval_data_loader, eval_y_list, device=device)

        for rest in tqdm(range(self.n_restarts)):
            print("restarting attack optimization, restart number: " + str(rest))
            opt_start_time = time.time()

            self.generator = self.generator.to(device)
            self.generator.train()

            for k in tqdm(range(self.n_iter)):
                print(f" attack optimization epoch: {str(k)} ")
                iter_start_time = time.time()

                self.gradient_ascent_step(data_shape, data_loader, y_list, clean_flow_list, eps, device=device)

                step_runtime = time.time() - iter_start_time
                print(f" optimization epoch finished, epoch runtime: {str(step_runtime)} ")

                print(" evaluating perturbation")
                eval_start_time = time.time()

                with torch.no_grad():
                    pert = self.generator.sample(device)
                    pert = self.project(pert, eps)
                    eval_loss_tot, eval_loss_list = self.attack_eval(pert, data_shape, eval_data_loader, eval_y_list,
                                                                     device)

                    if eval_loss_tot > best_loss_sum:
                        best_pert = pert.clone().detach()
                        best_loss_list = eval_loss_list
                        best_loss_sum = eval_loss_tot
                    all_loss.append(eval_loss_list)
                    all_best_loss.append(best_loss_list)
                    traj_loss_mean_list = np.mean(eval_loss_list, axis=0)
                    traj_best_loss_mean_list = np.mean(best_loss_list, axis=0)

                    eval_runtime = time.time() - eval_start_time
                    print(" evaluation finished, evaluation runtime: " + str(eval_runtime))
                    print(" current trajectories loss mean list:" + str(traj_loss_mean_list))
                    print(" current trajectories best loss mean list:" + str(traj_best_loss_mean_list))
                    print(" trajectories clean loss mean list:" + str(traj_clean_loss_mean_list))
                    print(" current trajectories loss sum:" + str(eval_loss_tot))
                    print(" current trajectories best loss sum:" + str(best_loss_sum))
                    print(" trajectories clean loss sum:" + str(clean_loss_sum))

                    pert_dir = 'pertubations/' + self.run_name
                    if not os.path.exists(pert_dir):
                        os.makedirs(pert_dir)
                    save_image(pert, pert_dir + '/pert_' + str(k) + '_' + str(eval_loss_tot) + '.png')

                    del eval_loss_tot
                    del eval_loss_list
                    torch.cuda.empty_cache()

            opt_runtime = time.time() - opt_start_time
            print("optimization restart finished, optimization runtime: " + str(opt_runtime))
        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss

