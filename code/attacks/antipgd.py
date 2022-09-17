import os

import numpy as np
import torch

from Datasets.tartanTrajFlowDataset import extract_traj_data
from attacks.attack import Attack
from attacks.pgd import PGD
import time

from torchvision.utils import save_image
from tqdm import tqdm
import cv2


class AntiPGD(PGD):
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
            noise=0.1,
    ):
        super(PGD, self).__init__(model, criterion, test_criterion, norm, data_shape,
                                  sample_window_size, sample_window_stride,
                                  pert_padding, run_name)

        self.alpha = alpha
        self.noise = noise
        self.n_restarts = n_restarts
        self.n_iter = n_iter

        self.rand_init = rand_init

        self.init_pert = None
        if init_pert_path is not None:
            self.init_pert = cv2.cvtColor(cv2.imread(init_pert_path), cv2.COLOR_BGR2RGB)
            if init_pert_transform is None:
                self.init_pert = torch.tensor(self.init_pert).unsqueeze(0)
            else:
                self.init_pert = init_pert_transform({'img': self.init_pert})['img'].unsqueeze(0)

        self.prev_noise = None

    def calc_sample_grad_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        pert = pert.detach()
        pert.requires_grad_()
        img1_adv, img2_adv, output_adv = self.perturb_model_single(pert, img1_I0, img2_I0,
                                                            intrinsic_I0,
                                                            img1_delta, img2_delta,
                                                            scale,
                                                            mask1, mask2,
                                                            perspective1,
                                                            perspective2,
                                                            device)
        loss = self.criterion(output_adv, scale.to(device), y.to(device), target_pose.to(device), clean_flow.to(device))
        loss_sum = loss.sum(dim=0)
        grad = torch.autograd.grad(loss_sum, [pert])[0].detach()

        del img1_adv
        del img2_adv
        del output_adv
        del loss
        del loss_sum
        torch.cuda.empty_cache()

        return grad

    def calc_sample_grad_split(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                         scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        sample_data_ind = list(range(img1_I0.shape[0] + 1))
        window_start_list = sample_data_ind[0::self.sample_window_stride]
        window_end_list = sample_data_ind[self.sample_window_size::self.sample_window_stride]

        if window_end_list[-1] != sample_data_ind[-1]:
            window_end_list.append(sample_data_ind[-1])
        grad = torch.zeros_like(pert, requires_grad=False)
        grad_multiplicity = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)

        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]
            grad_multiplicity[window_start:window_end] += 1

            pert_window = pert[window_start:window_end].clone().detach()
            img1_I0_window = img1_I0[window_start:window_end].clone().detach()
            img2_I0_window = img2_I0[window_start:window_end].clone().detach()
            intrinsic_I0_window = intrinsic_I0[window_start:window_end].clone().detach()
            img1_delta_window = img1_delta[window_start:window_end].clone().detach()
            img2_delta_window = img2_delta[window_start:window_end].clone().detach()
            scale_window = scale[window_start:window_end].clone().detach()
            y_window = y[window_start:window_end].clone().detach()
            clean_flow_window = clean_flow[window_start:window_end].clone().detach()
            target_pose_window = target_pose.clone().detach()
            perspective1_window = perspective1[window_start:window_end].clone().detach()
            perspective2_window = perspective2[window_start:window_end].clone().detach()
            mask1_window = mask1[window_start:window_end].clone().detach()
            mask2_window = mask2[window_start:window_end].clone().detach()

            grad_window = self.calc_sample_grad_single(pert_window,
                                                     img1_I0_window,
                                                     img2_I0_window,
                                                     intrinsic_I0_window,
                                                     img1_delta_window,
                                                     img2_delta_window,
                                                     scale_window,
                                                     y_window,
                                                     clean_flow_window,
                                                     target_pose_window,
                                                     perspective1_window,
                                                     perspective2_window,
                                                     mask1_window,
                                                     mask2_window,
                                                     device=device)
            with torch.no_grad():
                grad[window_start:window_end] += grad_window

            del grad_window
            del pert_window
            del img1_I0_window
            del img2_I0_window
            del intrinsic_I0_window
            del scale_window
            del y_window
            del clean_flow_window
            del target_pose_window
            del perspective1_window
            del perspective2_window
            del mask1_window
            del mask2_window
            torch.cuda.empty_cache()
        grad_multiplicity_expand = grad_multiplicity.view(-1, 1, 1, 1).expand(grad.shape)
        grad = grad / grad_multiplicity_expand
        del grad_multiplicity
        del grad_multiplicity_expand
        torch.cuda.empty_cache()
        return grad.to(device)


    def gradient_ascent_step(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                             multiplier, a_abs, eps, device=None):

        pert_expand = pert.expand(data_shape[0], -1, -1, -1).to(device)
        grad_tot = torch.zeros_like(pert, requires_grad=False)

        for data_idx, data in enumerate(data_loader):
            dataset_idx, dataset_name, traj_name, traj_len, \
            img1_I0, img2_I0, intrinsic_I0, \
            img1_I1, img2_I1, intrinsic_I1, \
            img1_delta, img2_delta, \
            motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
            mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
            grad = self.calc_sample_grad(pert_expand, img1_I0, img2_I0, intrinsic_I0,
                                         img1_delta, img2_delta,
                                         scale, y_list[data_idx], clean_flow_list[data_idx], patch_pose,
                                         perspective1, perspective2,
                                         mask1, mask2, device=device)
            grad = grad.sum(dim=0, keepdims=True).detach()

            with torch.no_grad():
                grad_tot += grad

            del grad
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
            torch.cuda.empty_cache()

        with torch.no_grad():

            sigma = self.noise

            grad = self.normalize_grad(grad_tot)
            noise = sigma * torch.rand(grad.shape, device=grad.device)
            if self.prev_noise is None:
                self.prev_noise = sigma * torch.rand(grad.shape, device=grad.device)
            print(f'noise norm: {torch.norm(noise - self.prev_noise)}')
            print(f'grad norm: {torch.norm(grad)}')
            print(f'pert norm: {torch.norm(pert)}')
            pert += multiplier * a_abs * grad + (noise - self.prev_noise)
            pert = self.project(pert, eps)

            self.prev_noise = noise
        return pert
