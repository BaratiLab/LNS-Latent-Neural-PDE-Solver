import torch
import torch.nn as nn
import os
import numpy as np
from einops import rearrange
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

        # load the data
        print('Loading data from', self.data_dir)
        with np.load(self.data_dir, mmap_mode='r') as data: # should be in .npz

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
                self.data_forward = data['all_sol_forward'][..., self.idxs]
                self.data_backward = data['all_sol_backward'][..., self.idxs]
                self.data_center = data['all_sol_center'][..., self.idxs]
            else:
                print('Using testing data')
                self.idxs = idxs[int(0.9 * len(idxs)):]
                self.data_center = data['all_sol_center'][..., self.idxs]

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
            return len(self.idxs) * self.case_len
        else:
            return len(self.idxs)

    def calculate_stats(self):
        self.stats = {
            'mean': np.mean(self.data_center),
            'std': np.std(self.data_center, axis=0).mean(),
        }

    def normalize_data(self, u):
        return (u - self.stats['mean']) / (self.stats['std'] + 1e-8)

    def __getitem__(self, idx):
        if self.train_mode:
            seed_to_read = idx // self.case_len
        else:
            seed_to_read = idx

        if self.train_mode:
            u_center = self.data_center[:, :, :, seed_to_read]
            u_forward = self.data_forward[:, :, :, seed_to_read]
            u_backward = self.data_backward[:, :, :, seed_to_read]
            input_t = idx % self.case_len
            # sample random snapshot
            x_center = u_center[input_t]  # [h, w]
            x_forward = u_forward[input_t]  # [h, w]
            x_backward = u_backward[input_t]  # [h, w]

            x_center = self.normalize_data(x_center)
            x_forward = self.normalize_data(x_forward)
            x_backward = self.normalize_data(x_backward)

            x_center = rearrange(torch.from_numpy(x_center).unsqueeze(-1), 'h w c -> c h w').float()
            x_forward = rearrange(torch.from_numpy(x_forward).unsqueeze(-1), 'h w c -> c h w').float()
            x_backward = rearrange(torch.from_numpy(x_backward).unsqueeze(-1), 'h w c -> c h w').float()
            return x_backward, x_center, x_forward
        else:
            u_center = self.data_center[:, :, :, seed_to_read]
            input_t = np.arange(self.case_len)
            x_center = u_center[input_t]  # [h, w]
            x_center = self.normalize_data(x_center)
            x_center = rearrange(torch.from_numpy(x_center).unsqueeze(-1), 't h w c -> t c h w').float()
            return x_center



    def denormalize(self,
                    x,  # [b, (t,) c, h, w]
                    ):
        # x: b h w c
        # impose Dirichlet boundary condition
        x = x.clone()
        # velocity
        x = x * self.stats['std'].item() + self.stats['mean'].item()

        return x