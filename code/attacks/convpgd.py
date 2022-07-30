import numpy as np
import torch
from Datasets.tartanTrajFlowDataset import extract_traj_data
from attacks.attack import Attack
import time

from torch.nn import ConvTranspose2d, Sequential
from tqdm import tqdm
import cv2


class PertGenerator(torch.nn.Module):

    '''
    this code is not clean, nor modular.
    '''
    def __init__(self, pert_shape):
        super(PertGenerator, self).__init__()
        self.pert_shape = pert_shape
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
        self.kernel = torch.ones(1, 1, 1, 1)
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
        c1 = ConvTranspose2d(1, 2, kernel_size=(7, 10), stride=(1, 1), padding=(0, 1))
        c2 = ConvTranspose2d(2, 3, kernel_size=(14, 25), stride=(2, 2), padding=(1, 2))
        c3 = ConvTranspose2d(3, 4, kernel_size=(35, 50), stride=(3, 3), padding=(4, 7))
        c4 = ConvTranspose2d(4, 3, kernel_size=(70, 100), stride=(4, 4), padding=(1, 4))
        self.cnn = Sequential(c1, c2, c3, c4)

        # c1 = ConvTranspose2d(1, 3, kernel_size=(448, 640), stride=(1, 1), padding=(0, 0))
        # self.cnn = Sequential(c1)


    def sample(self, device):
        return self.cnn(self.kernel.to(device))

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
            generator=None,
            generator_depth=3,

    ):
        super(ConvPGD, self).__init__(model, criterion, test_criterion, norm, data_shape,
                                  sample_window_size, sample_window_stride,
                                  pert_padding)

        self.alpha = alpha

        self.n_restarts = n_restarts
        self.n_iter = n_iter
        if generator is not None:
            self.generator = generator
        else:
            self.generator = PertGenerator(data_shape, depth=generator_depth)
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=alpha)

    #
    # def calc_sample_grad_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
    #                      scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
    #     pert = pert.detach()
    #     pert.requires_grad_()
    #     img1_adv, img2_adv, output_adv = self.perturb_model_single(pert, img1_I0, img2_I0,
    #                                                         intrinsic_I0,
    #                                                         img1_delta, img2_delta,
    #                                                         scale,
    #                                                         mask1, mask2,
    #                                                         perspective1,
    #                                                         perspective2,
    #                                                         device)
    #     loss = self.criterion(output_adv, scale.to(device), y.to(device), target_pose.to(device), clean_flow.to(device))
    #     loss_sum = loss.sum(dim=0)
    #     grad = torch.autograd.grad(loss_sum, [pert])[0].detach()
    #
    #     del img1_adv
    #     del img2_adv
    #     del output_adv
    #     del loss
    #     del loss_sum
    #     torch.cuda.empty_cache()
    #
    #     return grad
    #
    # def calc_sample_grad_split(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
    #                      scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
    #     sample_data_ind = list(range(img1_I0.shape[0] + 1))
    #     window_start_list = sample_data_ind[0::self.sample_window_stride]
    #     window_end_list = sample_data_ind[self.sample_window_size::self.sample_window_stride]
    #
    #     if window_end_list[-1] != sample_data_ind[-1]:
    #         window_end_list.append(sample_data_ind[-1])
    #     grad = torch.zeros_like(pert, requires_grad=False)
    #     grad_multiplicity = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
    #
    #     for window_idx, window_end in enumerate(window_end_list):
    #         window_start = window_start_list[window_idx]
    #         grad_multiplicity[window_start:window_end] += 1
    #
    #         pert_window = pert[window_start:window_end].clone().detach()
    #         img1_I0_window = img1_I0[window_start:window_end].clone().detach()
    #         img2_I0_window = img2_I0[window_start:window_end].clone().detach()
    #         intrinsic_I0_window = intrinsic_I0[window_start:window_end].clone().detach()
    #         img1_delta_window = img1_delta[window_start:window_end].clone().detach()
    #         img2_delta_window = img2_delta[window_start:window_end].clone().detach()
    #         scale_window = scale[window_start:window_end].clone().detach()
    #         y_window = y[window_start:window_end].clone().detach()
    #         clean_flow_window = clean_flow[window_start:window_end].clone().detach()
    #         target_pose_window = target_pose.clone().detach()
    #         perspective1_window = perspective1[window_start:window_end].clone().detach()
    #         perspective2_window = perspective2[window_start:window_end].clone().detach()
    #         mask1_window = mask1[window_start:window_end].clone().detach()
    #         mask2_window = mask2[window_start:window_end].clone().detach()
    #
    #         grad_window = self.calc_sample_grad_single(pert_window,
    #                                                  img1_I0_window,
    #                                                  img2_I0_window,
    #                                                  intrinsic_I0_window,
    #                                                  img1_delta_window,
    #                                                  img2_delta_window,
    #                                                  scale_window,
    #                                                  y_window,
    #                                                  clean_flow_window,
    #                                                  target_pose_window,
    #                                                  perspective1_window,
    #                                                  perspective2_window,
    #                                                  mask1_window,
    #                                                  mask2_window,
    #                                                  device=device)
    #         with torch.no_grad():
    #             grad[window_start:window_end] += grad_window
    #
    #         del grad_window
    #         del pert_window
    #         del img1_I0_window
    #         del img2_I0_window
    #         del intrinsic_I0_window
    #         del scale_window
    #         del y_window
    #         del clean_flow_window
    #         del target_pose_window
    #         del perspective1_window
    #         del perspective2_window
    #         del mask1_window
    #         del mask2_window
    #         torch.cuda.empty_cache()
    #     grad_multiplicity_expand = grad_multiplicity.view(-1, 1, 1, 1).expand(grad.shape)
    #     grad = grad / grad_multiplicity_expand
    #     del grad_multiplicity
    #     del grad_multiplicity_expand
    #     torch.cuda.empty_cache()
    #     return grad.to(device)

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

            # grad = self.calc_sample_grad(pert_expand, img1_I0, img2_I0, intrinsic_I0,
            #                              img1_delta, img2_delta,
            #                              scale, y_list[data_idx], clean_flow_list[data_idx], patch_pose,
            #                              perspective1, perspective2,
            #                              mask1, mask2, device=device)
            # grad = grad.sum(dim=0, keepdims=True).detach()

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
                print(" attack optimization epoch: " + str(k))
                iter_start_time = time.time()

                self.gradient_ascent_step(data_shape, data_loader, y_list, clean_flow_list, eps, device=device)

                step_runtime = time.time() - iter_start_time
                print(" optimization epoch finished, epoch runtime: " + str(step_runtime))

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
                    del eval_loss_tot
                    del eval_loss_list
                    torch.cuda.empty_cache()

            opt_runtime = time.time() - opt_start_time
            print("optimization restart finished, optimization runtime: " + str(opt_runtime))
        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss

