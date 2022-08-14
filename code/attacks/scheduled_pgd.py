import sys

import numpy as np
import torch
from Datasets.tartanTrajFlowDataset import extract_traj_data
from attacks.attack import Attack
import time

from torch.nn import ConvTranspose2d, Sequential
from tqdm import tqdm
import cv2

from attacks.pgd import PGD



class AlphaScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.n_iter = optimizer.n_iter
        self.alpha_init = optimizer.alpha
        self.cur_iter = 0

    def step(self):
        self.cur_iter += 1
        self.optimizer.alpha = self.alpha_init * (1 / self.cur_iter)
        return self.optimizer.alpha


#iter = 49
class NonScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        pass


class Factor_F_R_T:
    #iter=51
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.n_iter = optimizer.n_iter
        self.cur_iter = 0
        # self.initial_flow_factor = optimizer.criterion.flow_factor
        # self.initial_rot_factor = optimizer.criterion.rot_factor
        # self.initial_t_factor = optimizer.criterion.t_factor

        self.flow_list = [i for i in np.linspace(start=1.0, stop=0.0,num=self.n_iter//3)] + [0.0] * (self.n_iter - self.n_iter//3)
        self.rot_list = [i for i in np.linspace(start=0.0, stop=1.0,num=self.n_iter//3)] + [i for i in np.linspace(start=1.0,stop=0.0,num=self.n_iter//3)] + [0.0] * (self.n_iter - 2*self.n_iter//3)
        self.t_list = [0.0] * (self.n_iter - 2*self.n_iter//3) + [i for i in np.linspace(start=0.0,stop=1.0,num=self.n_iter//3)] + [1.0] * (self.n_iter - 2*self.n_iter//3)

        for i in range(self.n_iter):
            print(self.flow_list[i], self.rot_list[i], self.t_list[i])

    def step(self):
        self.cur_iter += 1
        self.optimizer.criterion.flow_factor = self.flow_list[self.cur_iter]
        self.optimizer.criterion.rot_factor = self.rot_list[self.cur_iter]
        self.optimizer.criterion.t_factor = self.t_list[self.cur_iter]
        print(self.optimizer.criterion.flow_factor, self.optimizer.criterion.rot_factor, self.optimizer.criterion.t_factor)



class SharpFactorScheduler:
    def __init__(self, optimizer, reverse=False):
        self.optimizer = optimizer
        self.n_iter = optimizer.n_iter
        self.alpha_init = optimizer.alpha
        self.cur_iter = 0

        self.initial_flow_factor = optimizer.criterion.flow_factor
        self.initial_rot_factor = optimizer.criterion.rot_factor
        self.initial_t_factor = optimizer.criterion.t_factor

        self.flow_list = [1.0] * (self.n_iter//3) + [0.0] * (self.n_iter - self.n_iter//3)
        self.rot_list = [0.0] * (self.n_iter//3) + [1.0] * (self.n_iter//3) + [0.0] * (self.n_iter - 2*self.n_iter//3)
        self.t_list = [0.0] * (2*self.n_iter//3) + [1.0] * (self.n_iter - 2*self.n_iter//3)

        if reverse:
            self.flow_list = self.flow_list[::-1]
            self.rot_list = self.rot_list[::-1]
            self.t_list = self.t_list[::-1]

    def step(self):
        self.optimizer.criterion.flow_factor = self.flow_list[self.cur_iter]
        self.optimizer.criterion.rot_factor = self.rot_list[self.cur_iter]
        self.optimizer.criterion.t_factor = self.t_list[self.cur_iter]
        self.cur_iter += 1



class Factor_T_R_F:
    #iter=52 - wrong factors but lets check this out...
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.n_iter = optimizer.n_iter
        self.cur_iter = 0
        # self.initial_flow_factor = optimizer.criterion.flow_factor
        # self.initial_rot_factor = optimizer.criterion.rot_factor
        # self.initial_t_factor = optimizer.criterion.t_factor

        self.t_list = [i for i in np.linspace(start=1.0, stop=0.0,num=self.n_iter//3)] + [1.0] * (self.n_iter - self.n_iter//3)
        self.rot_list = [i for i in np.linspace(start=0.0, stop=1.0,num=self.n_iter//3)] + [i for i in np.linspace(start=1.0,stop=0.0,num=self.n_iter//3)] + [0.0] * (self.n_iter - 2*self.n_iter//3)
        self.flow_list = [0.0] * (self.n_iter - 2*self.n_iter//3) + [i for i in np.linspace(start=0.0,stop=1.0,num=self.n_iter//3)] + [0.0] * (self.n_iter - 2*self.n_iter//3)

        for l in [self.t_list, self.rot_list, self.flow_list]:
            l += [0.0] * (self.n_iter - len(l))


        for i in range(self.n_iter):
            print(self.t_list[i], self.rot_list[i], self.flow_list[i])

    def step(self):
        self.cur_iter += 1
        self.optimizer.criterion.flow_factor = self.flow_list[self.cur_iter]
        self.optimizer.criterion.rot_factor = self.rot_list[self.cur_iter]
        self.optimizer.criterion.t_factor = self.t_list[self.cur_iter]
        print(self.optimizer.criterion.flow_factor, self.optimizer.criterion.rot_factor, self.optimizer.criterion.t_factor)





class ScheduledPGD(PGD):
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
            alpha_shceduler=None,
            factor_schedule=None,

    ):
        # alpha = 1.0  # this is because alpha is calculated inside calc_sample_grad
        super(ScheduledPGD, self).__init__(model, criterion, test_criterion, data_shape, norm, n_iter, n_restarts, alpha,
                                     rand_init, sample_window_size, sample_window_stride, pert_padding, init_pert_path,
                                     init_pert_transform)
        if alpha_shceduler is not None:
            self.alpha_shceduler = alpha_shceduler(self)
        else:
            self.alpha_shceduler = NonScheduler(self)  # <52|51> AlphaScheduler(self)

        if factor_schedule is not None:
            self.factor_schedule = factor_schedule(self)
        else:
            self.factor_schedule = SharpFactorScheduler(self, reverse=True) # Factor_T_R_F(self) # Factor_F_R_T(self)

        temp = SharpFactorScheduler(self, reverse=True)
        print("SharpFactorScheduler:")
        for i in range(temp.n_iter):
            temp.step()
            print(self.criterion.t_factor, self.criterion.rot_factor, self.criterion.flow_factor)

    def gradient_ascent_step(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                             multiplier, a_abs, eps, device=None):
        best_pert = super().gradient_ascent_step(pert, data_shape, data_loader, y_list, clean_flow_list, multiplier,
                                                 a_abs, eps, device)
        self.alpha_shceduler.step()
        self.factor_schedule.step()
        print("alpha:", self.alpha)
        return best_pert
