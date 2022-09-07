import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Datasets.tartanTrajFlowDataset import extract_traj_data
from attacks.pgd import PGD


class NoisePGD(PGD):
    # this class is like pgd, but with noise added to the input of the model
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
            noise=0.01):
        super(PGD, self).__init__(model=model, criterion=criterion, test_criterion=test_criterion, data_shape=data_shape, norm=norm, n_iter=n_iter, n_restarts=n_restarts, alpha=alpha, rand_init=rand_init, sample_window_size=sample_window_size, sample_window_stride=sample_window_stride, pert_padding=pert_padding, init_pert_path=init_pert_path, init_pert_transform=init_pert_transform, run_name=run_name)
        self.noise = noise
    def gradient_ascent_step(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                             multiplier, a_abs, eps, device=None):
            pert_expand = (pert + self.noise*torch.rand(pert.shape, device=device)).expand(data_shape[0], -1, -1, -1).to(device)
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
                pert += multiplier * a_abs * grad
                pert = self.project(pert, eps)

            return pert






