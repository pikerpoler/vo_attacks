import copy
import itertools
import os
import random
from pathlib import Path
import cv2
import torch
import matplotlib.pyplot as plt

from attacks.noisepgd import NoisePGD
from attacks import Const
from utils import get_args, compute_VO_args
import numpy as np
from Datasets.utils import plot_traj, visflow
from Datasets.transformation import ses2poses_quat, pos_quats2SE_matrices
from evaluator.tartanair_evaluator import TartanAirEvaluator
from os import mkdir
from os.path import isdir, join
from torchvision.utils import save_image
from Datasets.tartanTrajFlowDataset import extract_traj_data
from torch.utils.data import DataLoader
from random import sample
import gc
import csv


def save_flow_imgs(flow, flowdir, visflow, dataset_name, traj_name, save_dir_suffix='_clean'):
    flow_save_dir = flowdir + '/' + dataset_name
    if not isdir(flow_save_dir):
        mkdir(flow_save_dir)
    flow_save_dir = flow_save_dir + '/' + traj_name + save_dir_suffix
    if not isdir(flow_save_dir):
        mkdir(flow_save_dir)

    flow = flow.permute(0, 2, 3, 1).cpu().numpy()
    flowcount = 0
    for f in flow:
        np.save(flow_save_dir + '/' + str(flowcount).zfill(6) + '.npy', f)
        flow_vis = visflow(f)
        cv2.imwrite(flow_save_dir + '/' + str(flowcount).zfill(6) + '.png', flow_vis)
        flowcount += 1


def save_poses_se(motions, pose_quat_gt, pose_dir, plot_est=True, plot_gt=True, sc=0.05,
               dataset_name="dataset_dir", traj_name="traj_dir", save_dir_suffix='_clean'):

    pose_save_dir = pose_dir + '/' + dataset_name
    if not isdir(pose_save_dir):
        mkdir(pose_save_dir)
    pose_save_dir = pose_save_dir + '/' + traj_name + save_dir_suffix
    if not isdir(pose_save_dir):
        mkdir(pose_save_dir)
    print('==> saving poses to {}'.format(pose_save_dir))
    print()

    gt_poses = np.array(pos_quats2SE_matrices(pose_quat_gt))
    poses_quat = ses2poses_quat(motions.cpu().numpy())
    poses = np.array(pos_quats2SE_matrices(poses_quat))

    np.savetxt(pose_save_dir + '/est_poses_quat.txt', poses_quat)
    np.savetxt(pose_save_dir + '/gt_poses_quat.txt', pose_quat_gt)

    factor = 1
    alpha = 0.8
    fig_traj = plt.figure()
    ax = fig_traj.add_subplot(projection='3d')
    if plot_est:
        ax.scatter(poses[:, 0, -1], poses[:, 1, -1], poses[:, 2, -1], s=15, color=(1,0,1))
    if plot_gt:
        ax.scatter(gt_poses[:, 0, -1], gt_poses[:, 1, -1], gt_poses[:, 2, -1], s=15, color=(1,0,0,alpha))

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    zlims = ax.get_zlim()
    xlims = np.abs(xlims[1] - xlims[0])
    ylims = np.abs(ylims[1] - ylims[0])
    zlims = np.abs(zlims[1] - zlims[0])


    t_norm = []
    for p_idx, (p_gt, p) in enumerate(zip(gt_poses, poses)):
        tgt = p_gt[0:3, -1].reshape((3, 1))
        t = p[0:3, -1].reshape((3, 1))
        t_norm.append(np.linalg.norm(t-tgt))
        if p_idx % factor == 0:
            if plot_est:
                ax.plot([p[0, -1], p[0, -1] + sc * xlims * p[0, 0]], [p[1, -1], p[1, -1] + sc * ylims * p[1, 0]],
                        zs=[p[2, -1], p[2, -1] + sc * zlims * p[2, 0]], color=(1,0,1))
                ax.plot([p[0, -1], p[0, -1] + sc * xlims * p[0, 1]], [p[1, -1], p[1, -1] + sc * ylims * p[1, 1]],
                        zs=[p[2, -1], p[2, -1] + sc * zlims * p[2, 1]], color=(0,1,1))
                ax.plot([p[0, -1], p[0, -1] + sc * xlims * p[0, 2]], [p[1, -1], p[1, -1] + sc * ylims * p[1, 2]],
                        zs=[p[2, -1], p[2, -1] + sc * zlims * p[2, 2]], color=(0.5,0.5,0.5))
            if plot_gt:
                ax.plot([p_gt[0, -1], p_gt[0, -1] + sc * xlims * p_gt[0, 0]], [p_gt[1, -1], p_gt[1, -1] + sc * ylims * p_gt[1, 0]],
                        zs=[p_gt[2, -1], p_gt[2, -1] + sc * zlims * p_gt[2, 0]], color=(1,0,0,alpha))
                ax.plot([p_gt[0, -1], p_gt[0, -1] + sc * xlims * p_gt[0, 1]], [p_gt[1, -1], p_gt[1, -1] + sc * ylims * p_gt[1, 1]],
                        zs=[p_gt[2, -1], p_gt[2, -1] + sc * zlims * p_gt[2, 1]], color=(0,1,0,alpha))
                ax.plot([p_gt[0, -1], p_gt[0, -1] + sc * xlims * p_gt[0, 2]], [p_gt[1, -1], p_gt[1, -1] + sc * ylims * p_gt[1, 2]],
                        zs=[p_gt[2, -1], p_gt[2, -1] + sc * zlims * p_gt[2, 2]], color=(0,0,1,alpha))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.invert_zaxis()  # Since we use NED system
    plt.title(f'L2 final translation norm = {t_norm[-1]:.3f}[m]')
    plt.legend(labels=['estimated_x', 'estimated_y', 'estimated_z', 'GT_x', 'GT_y', 'GT_z'])
    ax.view_init(elev=89.99, azim=-90.)
    plt.savefig(join(pose_save_dir, 'xy_view.png'))
    ax.view_init(elev=180., azim=-90.)
    plt.savefig(join(pose_save_dir, 'xz_view.png'))
    ax.view_init(elev=-180., azim=0.)
    plt.savefig(join(pose_save_dir, 'yz_view.png'))
    ax.view_init(elev=20., azim=135)
    plt.savefig(join(pose_save_dir, 'view3d.png'))
    plt.close()


def save_img_tensors(img1, img2, path, names=None):
    if not isdir(path):
        mkdir(path)
    if names is None:
        imgs_num = len(img1)
        if img2 is not None:
            imgs_num += 1
        names = [str(idx) for idx in range(imgs_num)]
    for img_idx, img in enumerate(img1):
        save_image(img, path + '/' + names[img_idx] +'.png')
    if img2 is not None:
        save_image(img2[-1], path + '/' + names[-1] + '.png')


def save_unprocessed_imgs(img_dir, dataset_name, traj_name, img1_I0, img2_I0, img1_I1, img2_I1):
    img_save_dir = img_dir + '/' + dataset_name
    if not isdir(img_save_dir):
        mkdir(img_save_dir)

    save_img_tensors(img1_I0, img2_I0, img_save_dir + '/' + traj_name + '_I0')
    save_img_tensors(img1_I1, img2_I1, img_save_dir + '/' + traj_name + '_I1')


def save_adv_imgs(adv_img_dir, adv_pert_dir, dataset_name, traj_name, img1_adv, img2_adv, best_pert):
    adv_img_save_dir = adv_img_dir + '/' + dataset_name
    if not isdir(adv_img_save_dir):
        mkdir(adv_img_save_dir)
    adv_pert_save_dir = adv_pert_dir + '/' + dataset_name
    if not isdir(adv_pert_save_dir):
        mkdir(adv_pert_save_dir)
    save_img_tensors(img1_adv, img2_adv, adv_img_save_dir + '/' + traj_name)
    save_img_tensors(best_pert, best_pert, adv_pert_save_dir + '/' + traj_name)


def test_model(model, criterions, img1, img2, intrinsic, scale_gt, motions_target, target_pose=None,
               window_size=None, device=None):
    if window_size is None:
        if device is None:
            motions, flow = model.test_batch(img1, img2, intrinsic, scale_gt)
            crit_results = [crit((motions, flow), scale_gt, motions_target, target_pose).detach().cpu()
                            for crit in criterions]
            return (motions, flow), crit_results

        img1_device = img1.clone().detach().to(device)
        img2_device = img2.clone().detach().to(device)
        intrinsic_device = intrinsic.clone().detach().to(device)
        scale_gt_device = scale_gt.clone().detach().to(device)
        motions_target_device = motions_target.clone().detach().to(device)
        target_pose_device = target_pose.clone().detach().to(device)

        motions_device, flow_device = model.test_batch(img1_device, img2_device, intrinsic_device, scale_gt_device)
        crit_results_device = [crit((motions_device, flow_device), scale_gt_device,
                                    motions_target_device, target_pose_device)
                               for crit in criterions]
        motions = motions_device.clone().detach().cpu()
        flow = flow_device.clone().detach().cpu()
        crit_results = [crit_result_device.clone().detach().cpu() for crit_result_device in crit_results_device]

        del img1_device
        del img2_device
        del intrinsic_device
        del scale_gt_device
        del motions_target_device
        del target_pose_device
        del motions_device
        del flow_device
        del crit_results_device
        torch.cuda.empty_cache()

        return (motions, flow), crit_results

    data_ind = list(range(img1.shape[0] + 1))
    window_start_list = data_ind[0::window_size]
    window_end_list = data_ind[window_size::window_size]
    if window_end_list[-1] != data_ind[-1]:
        window_end_list.append(data_ind[-1])
    motions_window_list = []
    flow_window_list = []
    if device is None:
        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]

            img1_window = img1[window_start:window_end].clone().detach()
            img2_window = img2[window_start:window_end].clone().detach()
            intrinsic_window = intrinsic[window_start:window_end].clone().detach()
            scale_gt_window = scale_gt[window_start:window_end].clone().detach()

            motions_window, flow_window = model.test_batch(img1_window, img2_window, intrinsic_window, scale_gt_window)
            motions_window_list.append(motions_window)
            flow_window_list.append(flow_window)

            del img1_window
            del img2_window
            del intrinsic_window
            del scale_gt_window
            torch.cuda.empty_cache()

        motions = torch.cat(motions_window_list, dim=0)
        flow = torch.cat(flow_window_list, dim=0)
        crit_results = [crit((motions, flow), scale_gt, motions_target, target_pose).detach().cpu()
                        for crit in criterions]
        del motions_window_list
        del flow_window_list
        torch.cuda.empty_cache()
        return (motions, flow), crit_results

    for window_idx, window_end in enumerate(window_end_list):
        window_start = window_start_list[window_idx]

        img1_window = img1[window_start:window_end].clone().detach().to(device)
        img2_window = img2[window_start:window_end].clone().detach().to(device)
        intrinsic_window = intrinsic[window_start:window_end].clone().detach().to(device)
        scale_gt_window = scale_gt[window_start:window_end].clone().detach().to(device)

        motions_window, flow_window = model.test_batch(img1_window, img2_window, intrinsic_window, scale_gt_window)
        print("")
        print()
        motions_window_list.append(motions_window)
        flow_window_list.append(flow_window)

        del img1_window
        del img2_window
        del intrinsic_window
        del scale_gt_window
        torch.cuda.empty_cache()

    motions_device = torch.cat(motions_window_list, dim=0)
    flow_device = torch.cat(flow_window_list, dim=0)
    scale_gt_device = scale_gt.clone().detach().to(device)
    motions_target_device = motions_target.clone().detach().to(device)
    target_pose_device = target_pose.clone().detach().to(device)

    crit_results_device = [crit((motions_device, flow_device), scale_gt_device,
                                motions_target_device, target_pose_device)
                           for crit in criterions]
    motions = motions_device.clone().detach().cpu()
    flow = flow_device.clone().detach().cpu()
    crit_results = [crit_result_device.clone().detach().cpu() for crit_result_device in crit_results_device]

    del scale_gt_device
    del motions_target_device
    del target_pose_device
    del motions_device
    del flow_device
    del crit_results_device
    del motions_window_list
    del flow_window_list
    torch.cuda.empty_cache()

    return (motions, flow), crit_results


def test_clean_multi_inputs(args):
    dataset_idx_list = []
    dataset_name_list = []
    traj_name_list = []
    traj_indices = []
    motions_gt_list = []
    traj_clean_motions = []
    traj_motions_scales = []
    traj_mask_l0_ratio_list = []

    traj_clean_criterions_list = [[] for crit in args.criterions]
    frames_clean_criterions_list = [[[] for i in range(args.traj_len)] for crit in args.criterions]

    for traj_idx, traj_data in enumerate(args.testDataloader):
        dataset_idx, dataset_name, traj_name, traj_len, \
        img1_I0, img2_I0, intrinsic_I0, \
        img1_I1, img2_I1, intrinsic_I1, \
        img1_delta, img2_delta, \
        motions_gt, scale_gt, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(traj_data)

        traj_mask_l0_ratio = (mask.count_nonzero() / mask.numel()).item()

        dataset_idx_list.append(dataset_idx)
        dataset_name_list.append(dataset_name)
        traj_name_list.append(traj_name)
        traj_indices.append(traj_idx)
        motions_gt_list.append(motions_gt)
        traj_mask_l0_ratio_list.append(traj_mask_l0_ratio)

        if args.save_imgs:
            save_unprocessed_imgs(args.img_dir, dataset_name, traj_name, img1_I0, img2_I0, img1_I1, img2_I1)

        with torch.no_grad():
            (motions, flow), crit_results = \
                test_model(args.model, args.criterions,
                           img1_I0, img2_I0, intrinsic_I0, scale_gt, motions_gt, patch_pose,
                           window_size=args.window_size, device=args.device)
            for crit_idx, crit_result in enumerate(crit_results):
                crit_result_list = crit_result.tolist()
                traj_clean_criterions_list[crit_idx].append(crit_result_list)
                for frame_idx, frame_clean_crit in enumerate(crit_result_list):
                    frames_clean_criterions_list[crit_idx][frame_idx].append(frame_clean_crit)
            del crit_results

        if args.save_flow:
            save_flow_imgs(flow, args.flowdir, visflow, dataset_name, traj_name, save_dir_suffix='_clean')

        if args.save_pose:
            save_poses_se(motions, pose_quat_gt, args.pose_dir,
                       dataset_name=dataset_name, traj_name=traj_name, save_dir_suffix='_clean')

        traj_clean_motions.append(motions)
        traj_motions_scales.append(scale_gt.numpy())
        del flow
        del img1_I0
        del img2_I0
        del intrinsic_I0
        del img1_I1
        del img2_I1
        del intrinsic_I1
        del img1_delta
        del img2_delta
        del motions_gt
        del scale_gt
        del pose_quat_gt
        del mask
        del perspective
        torch.cuda.empty_cache()

    del frames_clean_criterions_list
    del traj_motions_scales
    torch.cuda.empty_cache()

    return dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, motions_gt_list, \
           traj_clean_criterions_list, traj_clean_motions


def test_adv_trajectories(dataloader, model, motions_target_list, attack, pert,
                          criterions, window_size,
                          save_imgs, save_flow, save_pose,
                          adv_img_dir, adv_pert_dir, flowdir, pose_dir,
                          device=None, multi_perturb=False):
    traj_adv_criterions_list = [[] for crit in criterions]
    for traj_idx, traj_data in enumerate(dataloader):
        dataset_idx, dataset_name, traj_name, traj_len, \
        img1_I0, img2_I0, intrinsic_I0, \
        img1_I1, img2_I1, intrinsic_I1, \
        img1_delta, img2_delta, \
        motions_gt, scale_gt, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(traj_data)
        motions_target = motions_target_list[traj_idx]
        traj_pert = pert
        if multi_perturb:
            traj_pert = pert[traj_idx]

        with torch.no_grad():
            img1_adv, img2_adv = attack.apply_pert(traj_pert, img1_I0, img2_I0, img1_delta, img2_delta,
                                                   mask, perspective, device)
            (motions_adv, flow_adv), crit_results = \
                test_model(model, criterions,
                           img1_adv, img2_adv, intrinsic_I0, scale_gt, motions_target, patch_pose,
                           window_size=window_size, device=device)
            for crit_idx, crit_result in enumerate(crit_results):
                traj_adv_criterions_list[crit_idx].append(crit_result.tolist())
            del crit_results


        if save_imgs:
            save_adv_imgs(adv_img_dir, adv_pert_dir, dataset_name, traj_name,
                          img1_adv,  img2_adv, traj_pert)

        if save_flow:
            save_flow_imgs(flow_adv, flowdir, visflow, dataset_name, traj_name, save_dir_suffix='_adv')

        if save_pose:
            save_poses_se(motions_adv, pose_quat_gt, pose_dir,
                       dataset_name=dataset_name, traj_name=traj_name, save_dir_suffix='_adv')

        del motions_adv
        del flow_adv
        del img1_I0
        del img2_I0
        del intrinsic_I0
        del img1_I1
        del img2_I1
        del intrinsic_I1
        del img1_delta
        del img2_delta
        del motions_gt
        del scale_gt
        del pose_quat_gt
        del patch_pose
        del mask
        del perspective
        del img1_adv
        del img2_adv
        torch.cuda.empty_cache()

    return traj_adv_criterions_list


def report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_crit_list, traj_adv_crit_list, save_csv, output_dir, crit_str,
                         experiment_name=""):
    traj_len = len(traj_clean_crit_list[0])
    csv_path = os.path.join(output_dir, 'results_' + crit_str + '.csv')
    frames_clean_crit_list = [[] for i in range(traj_len)]
    frames_adv_crit_list = [[] for i in range(traj_len)]
    frames_delta_crit_list = [[] for i in range(traj_len)]
    frames_ratio_crit_list = [[] for i in range(traj_len)]
    frames_delta_ratio_crit_list = [[] for i in range(traj_len)]
    traj_delta_crit_list = []
    traj_ratio_crit_list = []
    traj_delta_ratio_crit_list = []


    for traj_idx, traj_name in enumerate(traj_name_list):
        traj_clean_crit = traj_clean_crit_list[traj_idx]
        traj_adv_crit = traj_adv_crit_list[traj_idx]
        traj_delta_crit = []
        traj_ratio_crit = []
        traj_delta_ratio_crit = []

        for frame_idx, frame_clean_crit in enumerate(traj_clean_crit):
            frame_adv_crit = traj_adv_crit[frame_idx]
            frame_delta_crit = frame_adv_crit - frame_clean_crit
            if frame_clean_crit == 0:
                frame_ratio_crit = 0
                frame_delta_ratio_crit = 0
            else:
                frame_ratio_crit = frame_adv_crit / frame_clean_crit
                frame_delta_ratio_crit = frame_delta_crit / frame_clean_crit

            traj_delta_crit.append(frame_delta_crit)
            traj_ratio_crit.append(frame_ratio_crit)
            traj_delta_ratio_crit.append(frame_delta_ratio_crit)

            frames_clean_crit_list[frame_idx].append(frame_clean_crit)
            frames_adv_crit_list[frame_idx].append(frame_adv_crit)
            frames_delta_crit_list[frame_idx].append(frame_delta_crit)
            frames_ratio_crit_list[frame_idx].append(frame_ratio_crit)
            frames_delta_ratio_crit_list[frame_idx].append(frame_delta_ratio_crit)

        traj_delta_crit_list.append(traj_delta_crit)
        traj_ratio_crit_list.append(traj_ratio_crit)
        traj_delta_ratio_crit_list.append(traj_delta_ratio_crit)


    if save_csv:
        fieldsnames = ['dataset_idx', 'dataset_name', 'traj_idx', 'traj_name', 'frame_idx',
                       'clean_' + crit_str, 'adv_' + crit_str, 'adv_delta_' + crit_str,
                       'adv_ratio_' + crit_str, 'adv_delta_ratio_' + crit_str]

        with open(csv_path, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=fieldsnames)
            writer.writeheader()
            sum_adv_delta = 0
            num_samples = 0
            for traj_idx, traj_name in enumerate(traj_name_list):
                for frame_idx, frame_clean_crit in enumerate(traj_clean_crit_list[traj_idx]):
                    if frame_idx == 7:
                        sum_adv_delta += traj_delta_crit_list[traj_idx][frame_idx]
                        num_samples += 1
                    data = {'dataset_idx': dataset_idx_list[traj_idx],
                            'dataset_name': dataset_name_list[traj_idx],
                            'traj_idx': traj_indices[traj_idx],
                            'traj_name': traj_name,
                            'frame_idx': frame_idx,
                            'clean_' + crit_str: frame_clean_crit,
                            'adv_' + crit_str: traj_adv_crit_list[traj_idx][frame_idx],
                            'adv_delta_' + crit_str: traj_delta_crit_list[traj_idx][frame_idx],
                            'adv_ratio_' + crit_str: traj_ratio_crit_list[traj_idx][frame_idx],
                            'adv_delta_ratio_' + crit_str: traj_adv_crit_list[traj_idx][frame_idx]}
                    writer.writerow(data)
        avg_adv_delta = sum_adv_delta / num_samples
        print("Average " + crit_str + "_crit_adv_delta: " + str(avg_adv_delta))
        filename = os.path.join(output_dir, 'avarages_' + experiment_name + '.txt')
        with open(filename, 'a') as f:
            f.write("Average " + crit_str + "_crit_adv_delta: " + str(avg_adv_delta) + "\n")

    del traj_delta_crit_list
    del traj_ratio_crit_list
    del traj_adv_crit_list

    frames_clean_crit_mean = [np.mean(crit_list) for crit_list in frames_clean_crit_list]
    frames_clean_crit_std = [np.std(crit_list) for crit_list in frames_clean_crit_list]
    frames_adv_crit_mean = [np.mean(crit_list) for crit_list in frames_adv_crit_list]
    frames_adv_crit_std = [np.std(crit_list) for crit_list in frames_adv_crit_list]
    frames_delta_crit_mean = [np.mean(crit_list) for crit_list in frames_delta_crit_list]
    frames_delta_crit_std = [np.std(crit_list) for crit_list in frames_delta_crit_list]
    frames_ratio_crit_mean = [np.mean(crit_list) for crit_list in frames_ratio_crit_list]
    frames_ratio_crit_std = [np.std(crit_list) for crit_list in frames_ratio_crit_list]
    frames_delta_ratio_crit_mean = [np.mean(crit_list) for crit_list in frames_delta_ratio_crit_list]
    frames_delta_ratio_crit_std = [np.std(crit_list) for crit_list in frames_delta_ratio_crit_list]


    del frames_clean_crit_list
    del frames_adv_crit_list
    del frames_delta_crit_list
    del frames_ratio_crit_list
    del frames_delta_ratio_crit_list

    return frames_clean_crit_mean, frames_clean_crit_std, \
           frames_adv_crit_mean, frames_adv_crit_std, \
           frames_delta_crit_mean, frames_delta_crit_std, \
           frames_ratio_crit_mean, frames_ratio_crit_std, \
           frames_delta_ratio_crit_mean, frames_delta_ratio_crit_std

def run_attacks_train(args):
    print("Training and testing an adversarial perturbation on the whole dataset")
    print("A single single universal will be produced and then tested on the dataset")
    print("args.attack")
    print(args.attack)
    attack = args.attack_obj

    temp_args = copy.deepcopy(args)
    temp_args.testDataloader = temp_args.valDataloader
    _, _, _, _, eval_y_list, _, _ = test_clean_multi_inputs(temp_args)
    temp_args.testDataloader = None
    temp_args.valDataloader = None

    dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, \
    motions_gt_list, traj_clean_criterions_list, traj_clean_motions = \
        test_clean_multi_inputs(args)

    # print("traj_name_list")
    # print(traj_name_list)

    motions_target_list = motions_gt_list
    traj_clean_rms_list, traj_clean_mean_partial_rms_list, \
    traj_clean_target_rms_list, traj_clean_target_mean_partial_rms_list = tuple(traj_clean_criterions_list)


    best_pert, clean_loss_list, all_loss_list, all_best_loss_list = \
        attack.perturb(args.testDataloader, motions_target_list, eps=args.eps, device=args.device, eval_data_loader=args.valDataloader, eval_y_list=eval_y_list)


    torch.save(all_loss_list, os.path.join(args.output_dir, 'all_loss_list.pt'))
    components_listname = args.output_dir.split('/')

    # listname = components_listname[-5] + "_" + components_listname[-2] + "_" + components_listname[-1] + suffix + '.pt'  # + "-" + components_listname[-3]
    listname = ''
    for component in components_listname[:-1]:
        listname += '_' + component
    listname += '_' + components_listname[-1] + '.pt'
    listname = listname.split('opt_whole_trajectory')[-1]  # this is not elegant, but it works
    list_path = os.path.join("results/loss_lists", listname)
    if not isinstance(attack, Const):
        print(f'saving all_loss_list to {list_path}')
        torch.save(all_loss_list, list_path)


    if args.save_best_pert:
        pert_as_image = best_pert if len(best_pert.shape) == 3 else best_pert[0]
        save_image(pert_as_image, args.adv_best_pert_dir + '/' + 'adv_best_pert.png')

    traj_adv_criterions_list = \
        test_adv_trajectories(args.testDataloader, args.model, motions_target_list, attack, best_pert,
                              args.criterions, args.window_size,
                              args.save_imgs, args.save_flow, args.save_pose,
                              args.adv_img_dir, args.adv_pert_dir, args.flowdir, args.pose_dir,
                              device=args.device)
    traj_adv_rms_list, traj_adv_mean_partial_rms_list, \
    traj_adv_target_rms_list, traj_adv_target_mean_partial_rms_list = tuple(traj_adv_criterions_list)

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_target_mean_partial_rms_list, traj_adv_target_mean_partial_rms_list,
                         args.save_csv, args.output_dir, crit_str="target_mean_partial_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_target_rms_list, traj_adv_target_rms_list,
                         args.save_csv, args.output_dir, crit_str="target_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_mean_partial_rms_list, traj_adv_mean_partial_rms_list,
                         args.save_csv, args.output_dir, crit_str="mean_partial_rms")

    report_adv_deviation(dataset_idx_list, dataset_name_list, traj_name_list, traj_indices,
                         traj_clean_rms_list, traj_adv_rms_list,
                         args.save_csv, args.output_dir, crit_str="rms")


def run_attacks_train_silent(args):
    attack = args.attack_obj
    dataset_idx_list, dataset_name_list, traj_name_list, traj_indices, \
    motions_gt_list, traj_clean_criterions_list, traj_clean_motions = \
        test_clean_multi_inputs(args)

    motions_target_list = motions_gt_list
    traj_clean_rms_list, traj_clean_mean_partial_rms_list, \
    traj_clean_target_rms_list, traj_clean_target_mean_partial_rms_list = tuple(traj_clean_criterions_list)

    best_pert, clean_loss_list, all_loss_list, all_best_loss_list = \
        attack.perturb(args.testDataloader, motions_target_list, eps=args.eps, device=args.device)

    traj_adv_criterions_list = \
        test_adv_trajectories(args.testDataloader, args.model, motions_target_list, attack, best_pert,
                              args.criterions, args.window_size,
                              args.save_imgs, args.save_flow, args.save_pose,
                              args.adv_img_dir, args.adv_pert_dir, args.flowdir, args.pose_dir,
                              device=args.device)
    traj_adv_rms_list, traj_adv_mean_partial_rms_list, \
    traj_adv_target_rms_list, traj_adv_target_mean_partial_rms_list = tuple(traj_adv_criterions_list)


    sum_adv_delta = 0
    num_samples = 0
    for traj_idx, traj_name in enumerate(traj_name_list):
        for frame_idx, frame_clean_crit in enumerate(traj_clean_target_rms_list[traj_idx]):
            if frame_idx == 7:
                sum_adv_delta += traj_adv_target_rms_list[traj_idx][frame_idx]
                num_samples += 1

    avg_adv_delta = sum_adv_delta / num_samples
    return avg_adv_delta

def tune_lr_and_weights(args, alpha_list=(0.001, 0.005, 0.01, 0.05), t_factor_list=(1.0,), r_factor_list=(1.0,), f_factor_list=(1.0,)):
    losses = dict()
    num_iterations = len(alpha_list) * len(t_factor_list) * len(r_factor_list) * len(f_factor_list)
    num_epochs = args.attack_k
    print(f'num_iterations: {num_iterations}, num_epochs: {num_epochs}')
    print(f'estimated time for completion: {1.2 * num_iterations * num_epochs} minutes')
    for alpha in alpha_list:
        for t_factor in t_factor_list:
            for r_factor in r_factor_list:
                for f_factor in f_factor_list:
                    args.alpha = alpha
                    args.attack_t_factor = t_factor
                    args.attack_rot_factor = r_factor
                    args.attack_flow_factor = f_factor
                    args = compute_VO_args(args)
                    losses[(alpha, t_factor, r_factor, f_factor)] = run_attacks_train_silent(args)
                    print(f'alpha: {alpha}, t_factor: {t_factor}, r_factor: {r_factor}, f_factor: {f_factor}, loss: {losses[(alpha, t_factor, r_factor, f_factor)]}')
    losses = {k: v for k, v in sorted(losses.items(), key=lambda item: item[1])}
    print(losses)
    print("Best:", losses[max(losses, key=losses.get)])
    print("Worst:", losses[min(losses, key=losses.get)])


def test_clean(args):
    print("Testing the visual odometer on the I1 albedo image compared to the clean I0 albedo image, "
          "I1 is set as the adversarial perturbation")
    return run_attacks_train(args)


def main():
    args = get_args()
    if args.attack is None:
        return test_clean(args)
    return run_attacks_train(args)


if __name__ == '__main__':
    main()
