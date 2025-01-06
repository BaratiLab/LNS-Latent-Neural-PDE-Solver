import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision import utils as vutils
from modules.autoencoder2d_nonsquared import SimpleAutoencoder
import yaml
import shutil
from dataset.twophase_flow_stage1 import TankSloshingData
from utils import dict2namespace
import wandb
from training_utils import relative_lp_loss, GradientDomainLoss, prepare_training, log_images, log_sequence
from einops import rearrange


class TrainAE:
    def __init__(self, config):

        # check device
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.device = device

        self.autoencoder = SimpleAutoencoder(config).to(device=device)
        self.opt_ae = self.configure_optimizers(config)

        self.log_dir = config.log_dir

        # prepare wandb logging
        wandb.init(project=config.project_name,
                   config=config)
        self.train(config)

    def configure_optimizers(self, args):
        lr = args.learning_rate
        opt_vq = torch.optim.Adam(
            list(self.autoencoder.parameters()),
            lr=lr, eps=1e-08, betas=(args.beta1, args.beta2)
        )
        # print how many parameters
        # complex cound as 2 parameters
        total_params = []
        for p in self.autoencoder.parameters():
            if p.requires_grad:
                if p.is_complex():
                    total_params += [p.numel() * 2]
                else:
                    total_params += [p.numel()]
        print(f"Number of trainable parameters: {sum(total_params)}")

        return opt_vq

    def train(self, args):
        train_dataset = TankSloshingData(args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        steps_per_epoch = len(train_dataloader)
        for epoch in range(args.epochs):
            with tqdm(range(len(train_dataloader))) as pbar:
                if epoch % args.ckpt_every == 0:
                    self.validate_loop(args, epoch)
                    torch.save(self.autoencoder.state_dict(),
                               os.path.join(self.log_dir, "checkpoints", f"vqgan_epoch_{epoch}.pt"))
                for i, x in zip(pbar, train_dataloader):
                    x = x.to(self.device)

                    x_hat = self.autoencoder(x)
                    x_hat = train_dataset.denormalize(x_hat)
                    x = train_dataset.denormalize(x)
                    rec_loss = relative_lp_loss(x_hat, x, reduce_dim=(-1, -2), p=2, reduce_all=True)

                    ae_loss = rec_loss
                    wandb.log({
                        'Reconstruction Loss': rec_loss,
                    })
                    self.opt_ae.zero_grad()
                    ae_loss.backward()

                    self.opt_ae.step()
                    # self.opt_disc.step()

                    pbar.set_postfix(
                        VQ_Loss=np.round(ae_loss.cpu().detach().numpy().item(), 3),
                        EPOCH=epoch,
                    )

                    pbar.update(0)

        self.validate_loop(args, 'final')
        torch.save(self.autoencoder.state_dict(),
                   os.path.join(self.log_dir, "checkpoints", f"vqgan_epoch_final.pt"))
        wandb.finish()

    @torch.no_grad()
    def validate_loop(self, args, epoch_num):
        print('Testing')
        val_dataset = TankSloshingData(args, train_mode=False)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4)
        recon_loss_all = torch.zeros((len(val_dataset), args.case_len, args.in_channels), device=self.device)
        with tqdm(range(len(val_dataloader))) as pbar:
            for i, x in zip(pbar, val_dataloader):
                # will be in shape [b t c h w]
                x = x.to(device=self.device)
                x_hat = torch.zeros_like(x)

                for t in range(args.case_len):
                    x_hat[:, t] = self.autoencoder(x[:, t])
                x_hat = val_dataset.denormalize(x_hat)
                x = val_dataset.denormalize(x)

                recon_loss = relative_lp_loss(x_hat, x, reduce_dim=(-1, -2), p=2, reduce_all=False)
                if (i + 1) * 16 > len(val_dataset):
                    recon_loss_all[i * 16:] = recon_loss
                else:
                    recon_loss_all[i * 16:(i + 1) * 16] = recon_loss

                pbar.update(0)
        recon_loss = recon_loss_all.mean((0, 1))
        print(f'Validation Reconstruction Loss for vx: {recon_loss[0]}')
        print(f'Validation Reconstruction Loss for vy: {recon_loss[1]}')
        print(f'Validation Reconstruction Loss for pressure: {recon_loss[2]}')
        print(f'Validation Reconstruction Loss for vof: {recon_loss[3]}')


        # log some prediction
        log_sequence(x_hat[:, ::int(args.case_len//5), 0], os.path.join(self.log_dir, "samples", f"sample_{epoch_num}_vx.png"))
        log_sequence(x[:, ::(args.case_len//5), 0], os.path.join(self.log_dir, "samples", f"gt_{epoch_num}_vx.png"))
        log_sequence(x_hat[:, ::(args.case_len//5), 1], os.path.join(self.log_dir, "samples", f"sample_{epoch_num}_vy.png"))
        log_sequence(x[:, ::(args.case_len//5), 1], os.path.join(self.log_dir, "samples", f"gt_{epoch_num}_vy.png"))

        wandb.log({
            'val_recon_loss': recon_loss,
        })

        fig, ax = plt.subplots(figsize=[6, 4], dpi=200)
        # plot velocities' trend
        err = recon_loss_all.mean(dim=(0, 2)).squeeze().cpu().numpy()  # preserve time axis
        err_std = recon_loss_all.std(dim=0)[:, :2].mean().squeeze().cpu().numpy()

        # plot mean and std
        ax.plot(np.arange(0, args.case_len), err, color='b')
        ax.fill_between(np.arange(0, args.case_len), err - err_std, err + err_std, alpha=0.3, color='b')

        # Hide the right and top spines
        plt.ylabel(r'Relative $\mathcal{L}_2$ norm', fontsize=12)
        plt.xlabel('Timesteps', fontsize=12)
        plt.grid(which='both', linestyle='-.')

        wandb.log({'Validation error plot': wandb.Image(fig)})


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
    trainer = TrainAE(config)

    print('Running finished...')
    exit()



