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
from dataset.twophase_flow_stage2 import ConditionalTankSloshingData
from utils import dict2namespace
import wandb
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from modules.autoencoder2d_nonsquared import SimpleAutoencoder
from modules.basics import ResidualBlock, GroupNorm
from modules.cond_utils import zero_module, fourier_embedding
from training_utils import relative_lp_loss, log_sequence, prepare_training


class DilatedResidualBlock(nn.Module):
    def __init__(self,
                 dim,
                 cond_emb_dim,
                 dilation=1, padding_mode='circular'):
        super(DilatedResidualBlock, self).__init__()
        self.dim = dim
        self.dilation = dilation
        self.padding_mode = padding_mode

        self.cond_emb = nn.Linear(cond_emb_dim, dim)

        self.conv1 = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1,
                      padding_mode=self.padding_mode),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=self.dilation, dilation=self.dilation,
                      padding_mode=self.padding_mode),

        )

        self.cond_conv1 = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            nn.GELU(),
            zero_module(nn.Conv2d(self.dim, self.dim, kernel_size=(3, 3), padding=(1, 1),
                                  padding_mode=padding_mode)))

        self.cond_conv2 = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            nn.Conv2d(self.dim, self.dim, 1),
            nn.GELU(),
            zero_module(nn.Conv2d(self.dim, self.dim, 1)),
        )

        self.ffn = nn.Sequential(
            nn.GroupNorm(1, self.dim),
            nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=False),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 1, 1, 0, bias=False))

    def forward(self, x, param):
        emb_out = self.cond_emb(param)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        x_skip = x
        x = self.conv1(x)
        x = x + emb_out
        x = x_skip + self.cond_conv1(x)
        x = x + self.ffn(x * (1. + self.cond_conv2(emb_out)))
        return x


class SimpleCNN(nn.Module):
    def __init__(self,
                 latent_dim,  # dimension of the latent space
                 cond_emb_dim,  # dimension of the conditional embedding
                 prop_n_block,  # number of residual blocks in the propagation network
                 prop_n_embd,  # number of channels in the propagation network
                 dilation=2,
                 ):
        #
        super(SimpleCNN, self).__init__()
        self.latent_dim = latent_dim
        self.cond_emb_dim = cond_emb_dim
        self.prop_n_block = prop_n_block
        self.prop_n_embd = prop_n_embd

        self.in_proj = nn.Conv2d(self.latent_dim, self.prop_n_embd, 1, 1, 0)

        self.cond_emb_proj = nn.Sequential(
            nn.Linear(self.cond_emb_dim, self.cond_emb_dim),
            nn.GELU(),
            nn.Linear(self.cond_emb_dim, self.cond_emb_dim),
        )

        # n x resnet blocks
        self.net = nn.ModuleList(
                                 [DilatedResidualBlock(self.prop_n_embd,
                                                       cond_emb_dim=self.cond_emb_dim,
                                                       dilation=dilation,
                                                       padding_mode='zeros',
                                                       )
                                  for _ in range(self.prop_n_block)]
                                 )
        self.out_proj = nn.Sequential(
            GroupNorm(self.prop_n_embd),
            nn.Conv2d(self.prop_n_embd, self.latent_dim, 1, 1, 0))

    def forward(self, z, param):
        b, c, h, w = z.shape
        cond_emb = self.cond_emb_proj(fourier_embedding(param, dim=self.cond_emb_dim))
        z = self.in_proj(z)
        for layer in self.net:
            z = layer(z, cond_emb)
        z = self.out_proj(z)
        return z


class LatentDynamics(nn.Module):
    def __init__(self, args):
        super(LatentDynamics, self).__init__()

        self.ae = SimpleAutoencoder(args)

        self.latent_resolution = args.latent_resolution
        self.latent_dim = args.latent_dim

        self.propagator = SimpleCNN(
            latent_dim=self.latent_dim,
            cond_emb_dim=self.latent_dim,
            prop_n_block=args.prop_n_block,
            prop_n_embd=args.prop_n_embd,
            dilation=args.dilation,
        )

    def load_autoencoder(self, args):
        print("Loading pretrained autoencoder from {}".format(args.pretrained_checkpoint_path))
        self.ae.load_checkpoint(args.pretrained_checkpoint_path)
        print('Pretrained autoencoder loaded successfully')
        # set all the param not requiring gradients
        for param in self.ae.parameters():
            param.requires_grad = False
        self.ae.eval()

    @torch.no_grad()
    def x_to_z(self, x):
        z = self.ae.encode(x)
        return z

    @torch.no_grad()
    def z_to_x(self, z):
        x = self.ae.decode(z)
        return x

    def forward(self, z_in, z_out, param, loss_fn):
        # during training, use teacher forcing
        b, t_in = z_in.shape[:2]
        _, t_out = z_out.shape[:2]
        assert t_in == 1

        # rollout training
        z_pred = []
        z_in = z_in.squeeze()
        for t in range(t_out):
            z_new = self.propagator(z_in, param)
            z_pred.append(z_new)
            z_in = z_new
        z_pred = torch.stack(z_pred, dim=1)
        loss = loss_fn(z_pred, z_out)
        return loss

    def predict(self, x, steps, param, to_x=False):
        x = x.squeeze()
        z = self.x_to_z(x)
        out_lst = []
        z = z.squeeze()
        for t in range(steps):

            z_new = self.propagator(z, param)

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
        train_dataset = ConditionalTankSloshingData(args)
        train_dataset.encode_dataset(self.model.ae, device=self.device)
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
                for i, (z_in, z_out, param) in zip(pbar, train_dataloader):
                    optim.zero_grad()

                    z_in, z_out = z_in.to(self.device), z_out.to(self.device)
                    param = param.to(self.device)
                    loss = self.model(z_in, z_out, param, F.smooth_l1_loss)

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
        val_dataset = ConditionalTankSloshingData(args, train_mode=False)
        val_dataloader = DataLoader(val_dataset, batch_size=BS, shuffle=True, num_workers=4)
        frame_wise_rel_loss_all = torch.zeros((len(val_dataset),
                                               args.case_len - 1, args.in_channels),
                                              device=self.device)
        seq_wise_rel_loss_all = torch.zeros((len(val_dataset), args.in_channels), device=self.device)

        with tqdm(range(len(val_dataloader))) as pbar:
            for i, (x, y, param) in zip(pbar, val_dataloader):
                # y will be in shape [b t c h w]
                x, y = x.to(self.device), y.to(self.device)
                param = param.to(self.device)
                y_hat = self.model.predict(x, y.shape[1], param, to_x=True)
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
        print(f'Sequence Loss for vx: {seq_wise_rel_loss_all[:, 0].mean()}')
        print(f'Sequence Loss for vy: {seq_wise_rel_loss_all[:, 1].mean()}')
        print(f'Sequence Loss for pressure: {seq_wise_rel_loss_all[:, 2].mean()}')
        print(f'Sequence Loss for vof: {seq_wise_rel_loss_all[:, 3].mean()}')

        # log some prediction
        log_sequence(y_hat[:, ::int(args.case_len // 5), 0],
                     os.path.join(self.log_dir, "samples", f"sample_{epoch_num}_vx.png"))
        log_sequence(y[:, ::(args.case_len // 5), 0], os.path.join(self.log_dir, "samples", f"gt_{epoch_num}_vx.png"))
        log_sequence(y_hat[:, ::(args.case_len // 5), 1],
                     os.path.join(self.log_dir, "samples", f"sample_{epoch_num}_vy.png"))
        log_sequence(y[:, ::(args.case_len // 5), 1], os.path.join(self.log_dir, "samples", f"gt_{epoch_num}_vy.png"))

        wandb.log({
            'Sequence Loss for vx:': seq_wise_rel_loss_all[:, 0].mean(),
            'Sequence Loss for vy:': seq_wise_rel_loss_all[:, 1].mean(),
            'Sequence Loss for pressure:': seq_wise_rel_loss_all[:, 2].mean(),
            'Sequence Loss for vof:': seq_wise_rel_loss_all[:, 3].mean(),
        })

        fig, ax = plt.subplots(figsize=[6, 4], dpi=200)
        # plot velocities' trend
        err = frame_wise_rel_loss_all.mean(dim=(0, 2)).squeeze().cpu().numpy()  # preserve time axis
        err_std = frame_wise_rel_loss_all.std(dim=0)[:, :2].mean().squeeze().cpu().numpy()

        # plot mean and std
        ax.plot(np.arange(0, len(err)), err, color='b')
        ax.fill_between(np.arange(0, len(err)), err - err_std, err + err_std, alpha=0.3, color='b')

        # Hide the right and top spines
        plt.ylabel(r'Relative $\mathcal{L}_2$ norm', fontsize=12)
        plt.xlabel('Timesteps', fontsize=12)
        plt.grid(which='both', linestyle='-.')

        wandb.log({'Validation error plot': wandb.Image(fig)})

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



