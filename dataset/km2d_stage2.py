import torch
import torch.nn as nn
import os
import numpy as np
from einops import rearrange
import gc
from tqdm import tqdm


class KM2DData(torch.utils.data.Dataset):
    # will contain five fields,

    def __init__(self, config, train_mode=True):
        self.data_dir = config.data_dir
        self.resolution = config.resolution
        self.skip = 256 // self.resolution
        self.interval = config.interval

        self.case_len = config.case_len  # 30
        self.out_tw = config.out_tw  # 2
        print(f'Using sequence of length: {self.case_len}')

        self.train = train_mode
        self.total_num = config.train_num + config.test_num

        # this ensure same test set
        if self.train:
            self.idxs = list(range(config.train_num))
            self.seq_no = list(range(config.train_num))
        else:
            self.idxs = list(range(config.test_num))
            self.seq_no = list(range(self.total_num - config.test_num, self.total_num))  # count from the end

        self.stats = {}
        self.load_all_data(self.data_dir)
        if os.path.exists(config.dataset_stat):
            print('Loading dataset stats from', config.dataset_stat)
            stats = np.load(config.dataset_stat, allow_pickle=True)
            # npz to dict
            self.stats = {k: stats[k] for k in stats.files}
        else:
            print('Calculating dataset stats')
            self.prepare_data()
            print('Saving dataset stats to', config.dataset_stat)
            self.dump_stats(config.dataset_stat)
        print(f'Dataset stats: {self.stats}')

    def prepare_data(self):
        # load all training data and then calculate the mean/std online
        self.stats['vort_mean'] = self.data.mean()
        self.stats['vort_std'] = self.data.std(axis=1).mean()

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        # x: b n n n c, assume the order is prs, vel
        mean = torch.tensor(self.stats['vort_mean'], device=x.device, dtype=x.dtype)
        std = torch.tensor(self.stats['vort_std'], device=x.device, dtype=x.dtype)
        x = x * std + mean
        return x

    def dump_stats(self, f):
        np.savez(f, **self.stats)

    def load_all_data(self, path):
        data = np.load(path, mmap_mode='r')
        self.data = data[self.seq_no, :self.case_len, ::self.skip, ::self.skip]
        del data
        print('Loaded data from: ', path)
        print(f'Data shape: {self.data.shape}')

    def __len__(self):
        if self.train:
            return len(self.idxs) * (self.case_len - (self.out_tw+1)*self.interval)
        else:
            return len(self.idxs)

    @torch.no_grad()
    def encode_dataset(self, ae, device):
        # before second stage training, we can encode the data first to save up time
        # encode the data into indices
        print('Encoding data...')
        self.encoded_data = []
        for idx in tqdm(range(self.data.shape[0])):
            u = self.data[idx]
            u = (u - self.stats['vort_mean']) / self.stats['vort_std']
            u = rearrange(torch.from_numpy(u).unsqueeze(-1), 't h w c -> t c h w').float().to(device)
            z = ae.encode(u)  # [t, c, h, w]
            self.encoded_data.append(z.cpu().numpy())

    def __getitem__(self, idx):
        if self.train:
            seq_no = idx // (self.case_len - (self.out_tw + 1)*self.interval)
            start = np.random.randint(0, (self.case_len - (self.out_tw + 1)*self.interval))
        else:
            seq_no = idx
            start = 0

        if self.train:
            z = self.encoded_data[seq_no][start:start + (self.out_tw + 1)*self.interval: self.interval]
            z = torch.from_numpy(z).float()
            z_in = z[0:1]
            z_out = z[1:]
            return z_in, z_out

        else:
            # randomly sample a start time step
            feat = self.data[seq_no]

            feat = (feat - self.stats['vort_mean']) / self.stats['vort_std']
            feat_ = feat[start]
            feat_tsr = torch.from_numpy(feat_).float()
            feat_tsr = rearrange(feat_tsr, 'nx ny -> 1 nx ny')
            pred_ = feat[start+self.interval::self.interval]
            pred_tsr = torch.from_numpy(pred_).float()
            pred_tsr = rearrange(pred_tsr, 't nx ny -> t 1 nx ny')

            return feat_tsr, pred_tsr