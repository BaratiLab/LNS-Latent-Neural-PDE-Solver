import numpy as np
import torch
import os
import shutil
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch.nn.functional as F

def relative_lp_loss(pred, gt,
                     reduce_dim=(-1, -2, -3),
                     reduction='sum', eps=1e-8, reduce_all=False, p=2):
    reduce_fn = torch.mean if reduction == 'mean' else torch.sum

    gt_norm = reduce_fn((gt ** p), dim=reduce_dim)
    mask = gt_norm < eps
    gt_norm[mask] = eps
    diff = reduce_fn((pred - gt) ** p, dim=reduce_dim)
    diff = diff / gt_norm  # [b, c]
    if reduce_all:
        diff = diff.sqrt().mean()  # mean across channels and batch and any other dimensions
    else:
        diff = diff.sqrt()  # do nothing  [b, c]
    return diff


def pointwise_correlation(pred, gt, reduce_dim=(-1, -2, -3), eps=1e-8):
    pred_norm = torch.sqrt(torch.sum(pred ** 2, dim=reduce_dim, keepdim=True))
    gt_norm = torch.sqrt(torch.sum(gt ** 2, dim=reduce_dim, keepdim=True))
    pred_normalized = pred / (pred_norm + eps)
    gt_normalized = gt / (gt_norm + eps)
    corr = torch.sum(pred_normalized * gt_normalized, dim=reduce_dim)
    return corr



class GradientDomainLoss(torch.nn.Module):
    def __init__(self, weight_time, weight_space):
        super(GradientDomainLoss, self).__init__()
        self.weight_time = weight_time
        self.weight_space = weight_space

    def temporal_fd(self, x):
        # assume x is (B, C, T, H, W)
        # assert at least three time steps
        assert x.shape[2] >= 3, "Temporal FD requires at least three time steps"
        # compute finite difference, central diff
        # (B, C, T, H, W) -> (B, C, T-2, H, W)
        fd_t = x[:, :, 2:, :, :] - x[:, :, :-2, :, :]
        return fd_t

    def spatial_fd(self, x):
        # assume x is (B, C, T, H, W)
        # compute finite difference, central diff
        # (B, C, T, H, W) -> (B, C, T, H-2, W)
        fd_y = x[..., 2:, :] - x[..., :-2, :]
        # (B, C, T, H, W) -> (B, C, T, H, W-2)
        fd_x = x[..., :, 2:] - x[..., :, :-2]
        return fd_y, fd_x

    def forward(self, pred, gt):
        # TODO: hard coded for now
        # remove vof channel
        # b c t h w
        pred = pred[:, :-1, :, :]
        gt = gt[:, :-1, :, :]

        # t_fd_pred = self.temporal_fd(pred)
        # t_fd_gt = self.temporal_fd(gt)
        s_fd_pred_y, s_fd_pred_x = self.spatial_fd(pred)
        s_fd_gt_y, s_fd_gt_x = self.spatial_fd(gt)
        # compute loss
        # loss = self.weight_time * relative_lp_loss(t_fd_pred, t_fd_gt, reduce_dim=(-1, -2, -3), reduce_all=True) +\
        loss = self.weight_space * (
            relative_lp_loss(s_fd_pred_y, s_fd_gt_y, reduce_dim=(-1, -2), reduce_all=True, p=2) +
            relative_lp_loss(s_fd_pred_x, s_fd_gt_x, reduce_dim=(-1, -2), reduce_all=True, p=2))
        #relative_lp_loss(s_fd_pred_y, s_fd_gt_y, reduce_dim=(-1, -2), reduce_all=True, p=2)
        return loss


def prepare_training(log_dir, overwrite_exist):
    # if log_dir exists, delete it
    if os.path.exists(log_dir):
        if overwrite_exist:
            print('log_dir already exists, deleting it and creating a new one')
            shutil.rmtree(log_dir)
        else:
            raise Exception("log_dir already exists and overwrite argument is set to False,"
                            " please check your config file.")
    os.makedirs(log_dir)
    os.makedirs(os.path.join(log_dir, "checkpoints"))
    os.makedirs(os.path.join(log_dir, "samples"))
    os.makedirs(os.path.join(log_dir, "code_cache"))

    # cache the current file/ encoder.py / decoder.py / dataset.py
    shutil.copy(__file__, os.path.join(log_dir, "code_cache"))
    # copy all the stuff under dataset
    shutil.copytree(os.path.join(os.path.dirname(__file__), "dataset"), os.path.join(log_dir, "code_cache"), dirs_exist_ok=True)

    # copy all the stuff under modules
    shutil.copytree(os.path.join(os.path.dirname(__file__), "modules"), os.path.join(log_dir, "code_cache"), dirs_exist_ok=True)


def log_images(imgs, out_path):
    b, h, w = imgs.shape
    image = imgs.detach().cpu().numpy()
    image = image.reshape((b, h, w))
    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(b//4, 4),  # creates 2x2 grid of axes
                     )

    for ax, im_no in zip(grid, np.arange(b)):
        # Iterating over the grid returns the Axes.
        ax.imshow(
            image[im_no].reshape((h, w)),
            cmap='twilight',
        )

        ax.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def log_sequence(imgs, out_path):
    b, t, h, w = imgs.shape
    image = imgs.detach().cpu().numpy()
    image = image.reshape((b*t, h, w))
    fig = plt.figure(figsize=(8., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(b, t),  # creates 2x2 grid of axes
                     )

    for ax, im_no in zip(grid, np.arange(b*t)):
        # Iterating over the grid returns the Axes.
        ax.imshow(
            image[im_no].reshape((h, w)),
            cmap='twilight',
        )

        ax.axis('off')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()