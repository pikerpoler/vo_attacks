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


class DFSPGD(PGD):
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
            width=2,

    ):
        alpha = 1.0  # this is because alpha is calculated inside calc_sample_grad
        super(DFSPGD, self).__init__(model, criterion, test_criterion, data_shape, norm, n_iter, n_restarts, alpha,
                                     rand_init, sample_window_size, sample_window_stride, pert_padding, init_pert_path,
                                     init_pert_transform)
        self.alpha = alpha
        self.width = width
        self.best_i = 7
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

    def get_step_sizes(self):
        step_sizes = [1.0 / 2**self.best_i]
        for i in range(self.width + 1):
            step_sizes.append(1.0/2**(self.best_i + i))
            step_sizes.append(1.0/2**(self.best_i - i))
        return step_sizes

    # def test_pert(self, cur_pert, data_loader, y_list, device, verbose=False):
    #     if not verbose:
    #         save_stdout = sys.stdout
    #         sys.stdout = open('trash', 'w')
    #         res = super().test_pert(cur_pert, data_loader, y_list, device)
    #         sys.stdout = save_stdout
    #     else:
    #         res = super().test_pert(cur_pert, data_loader, y_list, device)
    #     return res

    def gradient_ascent_step(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                             multiplier, a_abs, eps, device=None, verbose=False):

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
            grad = self.normalize_grad(grad_tot)
            # we will chose the best a_abs for this step:
            best_loss = 0.0
            losses = {}

            def sum_loss_from_list2(list2):
                # list2 is losses per trajectory
                # list1 is loss per frame of a trajectory
                # returns the avarage for last frame loss.
                losses_per_frame = [list1[-1] for list1 in list2]
                # print(losses_per_frame)
                return sum(losses_per_frame)

            for i, a_abs in enumerate(self.get_step_sizes()):
                cur_pert = pert + a_abs * grad * multiplier
                cur_pert = self.project(cur_pert, eps)
                cur_loss, _ = self.attack_eval(cur_pert, data_shape, data_loader, y_list, device)
                losses[a_abs] = cur_loss
                if cur_loss > best_loss:
                    best_loss = cur_loss
                    best_pert = cur_pert
                    self.best_i = i
            print(f'losses: {losses}')
            print(f'best_loss: {best_loss}, best_i: {self.best_i}')
        return best_pert

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

            pert = torch.zeros_like(best_pert)

            if self.init_pert is not None:
                print(" perturbation initialized from provided image")
                pert = self.init_pert.to(best_pert)
            elif self.rand_init:
                print(" perturbation initialized randomly")
                pert = self.random_initialization(pert, eps)
            else:
                print(" perturbation initialized to zero")

            pert = self.project(pert, eps)

            for k in tqdm(range(self.n_iter)):
                print(" attack optimization epoch: " + str(k))
                iter_start_time = time.time()

                pert = self.gradient_ascent_step(pert, data_shape, data_loader, y_list, clean_flow_list,
                                                 multiplier, a_abs, eps, device=device)

                step_runtime = time.time() - iter_start_time
                print(" optimization epoch finished, epoch runtime: " + str(step_runtime))

                print(" evaluating perturbation")
                eval_start_time = time.time()

                with torch.no_grad():
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
                    del eval_loss_tot
                    del eval_loss_list
                    torch.cuda.empty_cache()

            opt_runtime = time.time() - opt_start_time
            print("optimization restart finished, optimization runtime: " + str(opt_runtime))
        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss
