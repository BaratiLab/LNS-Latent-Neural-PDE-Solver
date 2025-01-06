import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision import utils as vutils
from modules.autoencoder2d_half_periodic import SimpleAutoencoder
import yaml
import shutil
from dataset.Stage1_SW import SW2DData
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

        if config.resume_training:
            self.autoencoder.load_state_dict(torch.load(config.resume_ckpt))

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
        train_dataset = SW2DData(args, train_mode=True, load_all=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        steps_per_epoch = len(train_dataloader)
        for epoch in range(args.epochs):
            if epoch % args.ckpt_every == 0:
                    self.validate_loop(args, epoch)
                    torch.save(self.autoencoder.state_dict(),
                               os.path.join(self.log_dir, "checkpoints", f"vqgan_epoch_{epoch}.pt"))
            with tqdm(range(len(train_dataloader))) as pbar:
                for i, x_in in zip(pbar, train_dataloader):
                    x_in = x_in.to(self.device)
                    x_hat = self.autoencoder(x_in)
                    rec_loss = relative_lp_loss(x_hat, x_in, reduce_dim=(-1, -2), p=2, reduce_all=True)
                    vq_loss = rec_loss
                    wandb.log({
                        'Reconstruction Loss': rec_loss,
                    })
                    self.opt_ae.zero_grad()
                    vq_loss.backward()

                    self.opt_ae.step()
                    pbar.set_postfix(
                        AE_Loss=np.round(vq_loss.cpu().detach().numpy().item(), 3),
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
        val_dataset = SW2DData(args, train_mode=False, load_all=False)
        val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=1)
        recon_loss_all = torch.zeros((len(val_dataset), args.case_len-2, args.in_channels), device=self.device)

        with tqdm(range(len(val_dataloader))) as pbar:
            for i, x_in in zip(pbar, val_dataloader):
                # will be in shape [b t c h w]
                x_in = x_in.to(device=self.device)
                x_hat = torch.zeros_like(x_in)
                for t in range(x_in.shape[1]):
                    x_hat[:, t] = self.autoencoder(x_in[:, t])
                x_hat = val_dataset.denormalize(x_hat)
                x_in = val_dataset.denormalize(x_in)

                recon_loss = relative_lp_loss(x_hat, x_in, reduce_dim=(-1, -2), p=2, reduce_all=False)
                if (i + 1) * 20 > len(val_dataset):
                    recon_loss_all[i * 20:] = recon_loss
                else:
                    recon_loss_all[i * 20:(i + 1) * 20] = recon_loss

                pbar.update(0)

        # log some prediction
        log_sequence(x_hat[:, ::int(args.case_len//10), 0], os.path.join(self.log_dir, "samples", f"sample_vx_{epoch_num}.png"))
        log_sequence(x_in[:, ::int(args.case_len//10), 0], os.path.join(self.log_dir, "samples", f"gt_vx_{epoch_num}.png"))
        log_sequence(x_hat[:, ::int(args.case_len//10), 1], os.path.join(self.log_dir, "samples", f"sample_vy_{epoch_num}.png"))
        log_sequence(x_in[:, ::int(args.case_len//10), 1], os.path.join(self.log_dir, "samples", f"gt_vy_{epoch_num}.png"))
        log_sequence(x_hat[:, ::int(args.case_len//10), 2], os.path.join(self.log_dir, "samples", f"sample_prs_{epoch_num}.png"))
        log_sequence(x_in[:, ::int(args.case_len//10), 2], os.path.join(self.log_dir, "samples", f"gt_prs_{epoch_num}.png"))

        recon_loss = recon_loss_all.mean(0)
        print(f'Validation Reconstruction Loss on vx: {recon_loss[:, 0].mean()}')
        print(f'Validation Reconstruction Loss on vy: {recon_loss[:, 1].mean()}')
        print(f'Validation Reconstruction Loss on prs: {recon_loss[:, 2].mean()}')

        wandb.log({
            'Validation Reconstruction Loss on vx': recon_loss[:, 0].mean(),
            'Validation Reconstruction Loss on vy': recon_loss[:, 1].mean(),
            'Validation Reconstruction Loss on prs': recon_loss[:, 2].mean(),
        })


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



