import torch
import torch.nn as nn
import os
import numpy as np
from einops import rearrange
from tqdm import tqdm
import gc


class NS2DData(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 train_mode=True,  # return an array with points' value not sampled set to zero
                 ):
        self.data_dir = args.data_dir
        self.case_len = args.case_len
        self.dataset_stat = args.dataset_stat
        self.num_case = args.num_case

        self.in_tw = 1
        self.out_tw = args.out_tw
        self.interval = args.interval

        # load the data
        print('Loading data from', self.data_dir)
        # load the data
        print('Loading data from', self.data_dir)
        with np.load(self.data_dir, mmap_mode='r') as data:  # should be in .npz

            print('Data shape:', data['all_sol_center'].shape)
            idxs = np.arange(min(self.num_case, data['all_sol_center'].shape[-1]))
            np.random.seed(1)  # deterministic
            np.random.shuffle(idxs)

            self.train_mode = train_mode
            idxs = idxs[:]
            self.train_idxs = idxs[:int(0.9 * len(idxs))]
            if train_mode:
                print('Using training data')
                self.idxs = idxs[:int(0.9 * len(idxs))]
                self.data = data['all_sol_center'][..., self.idxs]
            else:
                print('Using testing data')
                self.idxs = idxs[int(0.9 * len(idxs)):]
                self.data = data['all_sol_center'][..., self.idxs]

            del data
            gc.collect()

        self.cache = {}
        if os.path.exists(self.dataset_stat):
            print('Loading dataset stats from', self.dataset_stat)
            stats = np.load(self.dataset_stat, allow_pickle=True)
            # npz to dict
            self.stats = {k: stats[k] for k in stats.files}
        else:
            print('Calculating dataset stats')
            self.calculate_stats()
            print('Saving dataset stats to', self.dataset_stat)
            np.savez(self.dataset_stat, **self.stats, allow_pickle=True)
        print(f'Dataset stats: {self.stats}')

    def __len__(self):
        if self.train_mode:
            if (self.in_tw + self.out_tw)*self.interval == self.case_len:
                return len(self.idxs)
            else:
                return len(self.idxs) * (self.case_len-(self.in_tw+self.out_tw)*self.interval)
        else:
            return len(self.idxs)

    def calculate_stats(self):
        self.stats = {
            'mean': np.mean(self.data),
            'std': np.std(self.data, axis=0).mean(),
        }

    def normalize_data(self, u):
        return (u - self.stats['mean']) / (self.stats['std'] + 1e-8)

    @torch.no_grad()
    def encode_dataset(self, vq_ae, device):
        # before second stage training, we can encode the data first to save up time
        # encode the data into indices
        print('Encoding data...')
        self.encoded_data = []
        for idx in tqdm(range(self.data.shape[-1])):
            u = self.data[..., idx]
            u = self.normalize_data(u)
            t = u.shape[0]
            u = rearrange(torch.from_numpy(u).unsqueeze(-1), 't h w c -> t c h w').float().to(device)
            _, codebook_indices, _ = vq_ae.encode(u)  # [t, h*w]
            self.encoded_data.append(codebook_indices.cpu().numpy().reshape(t, -1))

    def __getitem__(self, idx):

        if self.train_mode:
            if (self.in_tw + self.out_tw) * self.interval == self.case_len:
                seed_to_read = idx
            else:
                seed_to_read = idx // (self.case_len - (self.in_tw + self.out_tw) * self.interval)
        else:
            seed_to_read = idx

        u_all = self.data[..., seed_to_read]   # [t, h, w]

        # randomly sample a starting point
        if self.train_mode:
            if (self.in_tw + self.out_tw) * self.interval == self.case_len:
                start_t = 0
            else:
                start_t = 0 + idx % (self.case_len//self.interval - (self.in_tw+self.out_tw))
            z_idx_all = self.encoded_data[seed_to_read]  # [t, h*w]
            z_idx_all = z_idx_all[:self.case_len:self.interval]  # [t, h*w]
            z_idx_in = z_idx_all[start_t:start_t + self.in_tw]  # [tin, h*w]
        else:
            start_t = 0
        u_all = u_all[:self.case_len:self.interval, None, ...]
        x = u_all[start_t:start_t+self.in_tw, 0]  # [t, c, h, w]
        x = self.normalize_data(x)
        x = torch.from_numpy(x).float()  # [t, c, h, w]

        if self.train_mode:
            output_t = start_t + self.in_tw
            y = u_all[output_t:output_t+self.out_tw]  # [c, h, w]
            z_idx_out = z_idx_all[output_t:output_t+self.out_tw]  # [t, h*w]
            # normalize the output
            y = self.normalize_data(y)
            y = torch.from_numpy(y).float()  # [t, c, h, w]
            return x, y, z_idx_in, z_idx_out
        else:
            output_t = start_t + self.in_tw
            y = u_all[output_t:]  # [t, c, h, w]
            # normalize the output
            y = self.normalize_data(y)
            y = torch.from_numpy(y).float()

            return x, y   # don't need z_idx_in and z_idx_out for testing

    def denormalize(self,
                    x,  # [b, (t,) c, h, w]
                    ):
        # x: b h w c
        # impose Dirichlet boundary condition
        x = x.clone()
        # velocity
        x = x * self.stats['std'].item() + self.stats['mean'].item()

        return x


class SimpleNS2DData(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 train_mode=True,  # return an array with points' value not sampled set to zero
                 ):
        self.data_dir = args.data_dir
        self.case_len = args.case_len
        self.dataset_stat = args.dataset_stat
        self.num_case = args.num_case

        self.in_tw = 1
        self.out_tw = args.out_tw
        self.interval = args.interval

        # load the data
        print('Loading data from', self.data_dir)
        # load the data
        print('Loading data from', self.data_dir)
        with np.load(self.data_dir, mmap_mode='r') as data:  # should be in .npz

            print('Data shape:', data['all_sol_center'].shape)
            idxs = np.arange(min(self.num_case, data['all_sol_center'].shape[-1]))
            np.random.seed(1)  # deterministic
            np.random.shuffle(idxs)

            self.train_mode = train_mode
            idxs = idxs[:]
            self.train_idxs = idxs[:int(0.9 * len(idxs))]
            if train_mode:
                print('Using training data')
                self.idxs = idxs[:int(0.9 * len(idxs))]
                self.data = data['all_sol_center'][..., self.idxs]
            else:
                print('Using testing data')
                self.idxs = idxs[int(0.9 * len(idxs)):]
                self.data = data['all_sol_center'][..., self.idxs]

            del data
            gc.collect()

        self.cache = {}
        if os.path.exists(self.dataset_stat):
            print('Loading dataset stats from', self.dataset_stat)
            stats = np.load(self.dataset_stat, allow_pickle=True)
            # npz to dict
            self.stats = {k: stats[k] for k in stats.files}
        else:
            print('Calculating dataset stats')
            self.calculate_stats()
            print('Saving dataset stats to', self.dataset_stat)
            np.savez(self.dataset_stat, **self.stats, allow_pickle=True)
        print(f'Dataset stats: {self.stats}')

    def __len__(self):
        if self.train_mode:
            if (self.in_tw + self.out_tw)*self.interval == self.case_len:
                return len(self.idxs)
            else:
                return len(self.idxs) * (self.case_len-(self.in_tw+self.out_tw)*self.interval)
        else:
            return len(self.idxs)

    def calculate_stats(self):
        self.stats = {
            'mean': np.mean(self.data),
            'std': np.std(self.data, axis=0).mean(),
        }

    def normalize_data(self, u):
        return (u - self.stats['mean']) / (self.stats['std'] + 1e-8)

    def __getitem__(self, idx):

        if self.train_mode:
            if (self.in_tw + self.out_tw) * self.interval == self.case_len:
                seed_to_read = idx
            else:
                seed_to_read = idx // (self.case_len - (self.in_tw + self.out_tw) * self.interval)
        else:
            seed_to_read = idx

        u_all = self.data[..., seed_to_read]   # [t, h, w]

        # randomly sample a starting point
        if self.train_mode:
            if (self.in_tw + self.out_tw) * self.interval == self.case_len:
                start_t = 0
            else:
                start_t = 0 + idx % (self.case_len//self.interval - (self.in_tw+self.out_tw))
        else:
            start_t = 0
        u_all = u_all[:self.case_len:self.interval, None, ...]
        x = u_all[start_t:start_t+self.in_tw, 0]  # [t, c, h, w]
        x = self.normalize_data(x)
        x = torch.from_numpy(x).float()  # [t, c, h, w]

        if self.train_mode:
            output_t = start_t + self.in_tw
            y = u_all[output_t:output_t+self.out_tw]  # [c, h, w]
            # normalize the output
            y = self.normalize_data(y)
            y = torch.from_numpy(y).float()  # [t, c, h, w]
            return x, y
        else:
            output_t = start_t + self.in_tw
            y = u_all[output_t:]  # [t, c, h, w]
            # normalize the output
            y = self.normalize_data(y)
            y = torch.from_numpy(y).float()

            return x, y   # don't need z_idx_in and z_idx_out for testing

    def denormalize(self,
                    x,  # [b, (t,) c, h, w]
                    ):
        # x: b h w c
        # impose Dirichlet boundary condition
        x = x.clone()
        # velocity
        x = x * self.stats['std'].item() + self.stats['mean'].item()

        return x