import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange
from torch.optim.lr_scheduler import OneCycleLR, StepLR, CosineAnnealingLR
import os


import yaml
import shutil
from dataset.Stage2_SW import SW2DData
from utils import dict2namespace
import wandb
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from modules.autoencoder2d_half_periodic import SimpleAutoencoder, HalfPeriodicConv2d,  HalfPeriodicResBlock2d
from modules.basics import GroupNorm
from training_utils import relative_lp_loss, log_sequence, prepare_training


class DilatedResidualBlock(nn.Module):
    def __init__(self, dim, dilation=1, periodic_direction='x'):
        super(DilatedResidualBlock, self).__init__()
        self.dim = dim
        self.dilation = dilation

        self.conv = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            HalfPeriodicConv2d(self.dim, self.dim, kernel_size=3, stride=1,
                               padding=1, periodic_direction=periodic_direction),
            nn.GELU(),
            HalfPeriodicConv2d(
                self.dim, self.dim, kernel_size=3, stride=1, padding=self.dilation,
                dilation=self.dilation, periodic_direction=periodic_direction
            ),
            nn.GELU(),
            HalfPeriodicConv2d(self.dim, self.dim, kernel_size=3, stride=1,
                               padding=1, periodic_direction=periodic_direction),
        )

        self.ffn = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=False))

    def forward(self, x):
        x = x + self.conv(x)
        x = x + self.ffn(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self,
                 latent_dim,  # dimension of the latent space
                 prop_n_block,  # number of residual blocks in the propagation network
                 prop_n_embd,  # number of channels in the propagation network
                 dilation=2,
                 ):
        #
        super(SimpleCNN, self).__init__()
        self.latent_dim = latent_dim
        self.prop_n_block = prop_n_block
        self.prop_n_embd = prop_n_embd

        self.in_proj = nn.Conv2d(self.latent_dim, self.prop_n_embd, 1, 1, 0)

        # n x resnet blocks
        self.net = nn.Sequential(*
                                 [DilatedResidualBlock(self.prop_n_embd,
                                                       dilation=dilation,
                                                       periodic_direction='x')
                                  for _ in range(self.prop_n_block)]
                                 )
        self.out_proj = nn.Sequential(
            GroupNorm(self.prop_n_embd),
            nn.Conv2d(self.prop_n_embd, self.latent_dim, 1, 1, 0))

    def forward(self, z):
        b, c, h, w = z.shape
        z = self.in_proj(z)
        z = self.net(z)   # spatial mixing
        z = self.out_proj(z)
        return z


class LatentDynamics(nn.Module):
    def __init__(self, args):
        super(LatentDynamics, self).__init__()

        self.vq_ae = SimpleAutoencoder(args)

        self.latent_resolution = args.latent_resolution
        self.latent_dim = args.latent_dim

        self.propagator = SimpleCNN(
            latent_dim=self.latent_dim,
            prop_n_block=args.prop_n_block,
            prop_n_embd=args.prop_n_embd,
            dilation=args.dilation,
        )

    def load_autoencoder(self, args):
        print("Loading pretrained autoencoder from {}".format(args.pretrained_checkpoint_path))
        self.vq_ae.load_checkpoint(args.pretrained_checkpoint_path, device=args.device)
        print('Pretrained autoencoder loaded successfully')
        # set all the param not requiring gradients
        for param in self.vq_ae.parameters():
            param.requires_grad = False
        self.vq_ae.eval()

    @torch.no_grad()
    def x_to_z(self, x):
        z = self.vq_ae.encode(x)
        return z

    @torch.no_grad()
    def z_to_x(self, z):
        x = self.vq_ae.decode(z)
        return x

    def forward(self, z_in, z_out, loss_fn):
        # during training, use teacher forcing
        b, t_in = z_in.shape[:2]
        _, t_out = z_out.shape[:2]
        assert t_in == 1

        # rollout training
        z_pred = []
        z_in = z_in.squeeze()
        for t in range(t_out):
            z_new = self.propagator(z_in)
            z_pred.append(z_new)
            z_in = z_new
        z_pred = torch.stack(z_pred, dim=1)
        loss = loss_fn(z_pred, z_out)
        return loss

    def predict(self, x, steps, to_x=False):
        x = x.squeeze()
        z = self.x_to_z(x)
        out_lst = []
        z = z.squeeze()
        for t in range(steps):

            z_new = self.propagator(z)

            z = z_new
            if to_x:
                y_hat = self.z_to_x(z_new)
                out_lst.append(y_hat)
            else:
                out_lst.append(z_new)
        out_lst = torch.stack(out_lst, dim=1)
        return out_lst


class TrainDynamics:
    def __init__(self, args):
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = LatentDynamics(args).to(device=device)
        self.model.load_autoencoder(args)
        self.log_dir = args.log_dir

        # prepare wandb logging
        wandb.init(project=args.project_name,
                   config=args)
        self.train(args)

    def configure_optimizers(self, args, dataloader):

        optimizer = torch.optim.Adam([p for p in self.model.propagator.parameters() if p.requires_grad],
                                     lr=args.learning_rate)

        print("Number of parameters: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        # use lr scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        return optimizer, scheduler

    def train(self, args):
        train_dataset = SW2DData(args)
        train_dataset.encode_dataset(self.model.vq_ae, device=self.device)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size, shuffle=True,
                                      num_workers=4, drop_last=True)
        optim, sched = self.configure_optimizers(args, train_dataloader)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataloader))) as pbar:
                if epoch % args.ckpt_every == 0:
                    self.validate_loop(args, epoch)
                    # save model
                    torch.save(self.model.state_dict(), os.path.join(self.log_dir,
                                                                     'checkpoints', f"model_{epoch}.pt"))
                    torch.save(optim.state_dict(), os.path.join(self.log_dir,
                                                                     'checkpoints', f"optim_{epoch}.pt"))
                    torch.save(sched.state_dict(), os.path.join(self.log_dir,
                                                                        'checkpoints', f"sched_{epoch}.pt"))
                for i, (z_in, z_out) in zip(pbar, train_dataloader):
                    optim.zero_grad()

                    z_in, z_out = z_in.to(self.device), z_out.to(self.device)
                    loss = self.model(z_in, z_out, F.smooth_l1_loss)

                    loss.backward()
                    optim.step()

                    pbar.set_postfix(
                        epoch=epoch,
                        pred_loss=np.round(loss.cpu().detach().numpy().item(), 4),
                        LR=np.round(optim.param_groups[0]['lr'], 6)
                    )
                    wandb.log({
                        'loss': loss,
                    })
                    pbar.update(0)
                sched.step()

        self.validate_loop(args, epoch)
        # save model
        torch.save(self.model.state_dict(), os.path.join(self.log_dir, f"model_final.pt"))
        torch.save(optim.state_dict(), os.path.join(self.log_dir,
                                                    'checkpoints', f"optim_final.pt"))
        torch.save(sched.state_dict(), os.path.join(self.log_dir,
                                                        'checkpoints', f"sched_final.pt"))
        wandb.finish()

    @torch.no_grad()
    def validate_loop(self, args, epoch_num):
        print('Testing')
        BS = 10
        val_dataset = SW2DData(args, train_mode=False)
        val_dataloader = DataLoader(val_dataset, batch_size=BS, shuffle=True, num_workers=4)
        frame_wise_rel_loss_all = torch.zeros((len(val_dataset),
                                               (args.case_len - 2)//args.interval-1, args.in_channels),
                                              device=self.device)
        seq_wise_rel_loss_all = torch.zeros((len(val_dataset), args.in_channels), device=self.device)

        with tqdm(range(len(val_dataloader))) as pbar:
            for i, (x, y) in zip(pbar, val_dataloader):
                # y will be in shape [b t c h w]
                x, y = x.to(self.device), y.to(self.device)
                y_hat = self.model.predict(x, y.shape[1], to_x=True)
                y_hat = val_dataset.denormalize(y_hat)
                y = val_dataset.denormalize(y)
                frame_wise_rel_loss = relative_lp_loss(y_hat, y, reduce_dim=(3, 4), p=2, reduce_all=False)
                seq_wise_rel_loss = relative_lp_loss(y_hat, y, reduce_dim=(1, 3, 4), p=2, reduce_all=False)
                if (i+1)*BS > len(val_dataset):
                    frame_wise_rel_loss_all[i*BS:] = frame_wise_rel_loss
                    seq_wise_rel_loss_all[i*BS:] = seq_wise_rel_loss
                else:
                    frame_wise_rel_loss_all[i*BS:(i+1)*BS] = frame_wise_rel_loss
                    seq_wise_rel_loss_all[i*BS:(i+1)*BS] = seq_wise_rel_loss

                pbar.update(0)
        # log some prediction
        log_sequence(y_hat[:, ::int(args.case_len // 5), 0],
                     os.path.join(self.log_dir, "samples", f"sample_vx_{epoch_num}.png"))
        log_sequence(y[:, ::int(args.case_len // 5), 0],
                     os.path.join(self.log_dir, "samples", f"gt_vx_{epoch_num}.png"))
        log_sequence(y_hat[:, ::int(args.case_len // 5), 1],
                     os.path.join(self.log_dir, "samples", f"sample_vy_{epoch_num}.png"))
        log_sequence(y[:, ::int(args.case_len // 5), 1],
                     os.path.join(self.log_dir, "samples", f"gt_vy_{epoch_num}.png"))
        log_sequence(y_hat[:, ::int(args.case_len // 5), 2],
                     os.path.join(self.log_dir, "samples", f"sample_prs_{epoch_num}.png"))
        log_sequence(y[:, ::int(args.case_len // 5), 2],
                     os.path.join(self.log_dir, "samples", f"gt_prs_{epoch_num}.png"))

        pred_loss = seq_wise_rel_loss_all.mean(0)
        print(f'Validation Prediction Loss on vx: {pred_loss[0]}')
        print(f'Validation Prediction Loss on vy: {pred_loss[1]}')
        print(f'Validation Prediction Loss on prs: {pred_loss[2]}')

        wandb.log({
            'val_pred_loss_vx': pred_loss[0],
            'val_pred_loss_vy': pred_loss[1],
            'val_pred_loss_prs': pred_loss[2],
        })

        def create_error_plot(err, err_std):

            fig, ax = plt.subplots(figsize=[6, 4], dpi=200)
            # plot velocities' trend

            # plot mean and std
            ax.plot(np.arange(len(err)), err, color='b')
            ax.fill_between(np.arange(len(err)), err - err_std, err + err_std, alpha=0.3, color='b')

            # Hide the right and top spines
            plt.ylabel(r'Relative $\mathcal{L}_2$ norm', fontsize=12)
            plt.xlabel('Timesteps', fontsize=12)
            plt.grid(which='both', linestyle='-.')
            return fig

        fig_field = create_error_plot(frame_wise_rel_loss_all.mean(dim=0)[:, -1].cpu().numpy(),
                                      frame_wise_rel_loss_all.std(dim=0)[:, -1].cpu().numpy())
        fig_vel = create_error_plot(frame_wise_rel_loss_all.mean(dim=0)[:, :2].mean(-1).cpu().numpy(),
                                    frame_wise_rel_loss_all.std(dim=0)[:, :2].mean(-1).cpu().numpy())
        wandb.log({'Validation error plot of velocity': wandb.Image(fig_vel)})
        wandb.log({'Validation error plot of field': wandb.Image(fig_field)})

        plt.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--comment', type=str, default='', help='Comment')
    args = parser.parse_args()

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)
    # copy the config file to the log_dir
    prepare_training(config.log_dir, config.overwrite_exist)
    shutil.copy(args.config, os.path.join(config.log_dir, 'config.yaml'))
    return args, config


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args, config = parse_args_and_config()
    set_random_seed(args.seed)
    # create the trainer
    train_dynamics = TrainDynamics(config)

    print('Running finished...')
    exit()



