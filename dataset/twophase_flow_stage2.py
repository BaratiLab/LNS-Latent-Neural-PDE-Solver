import torch
import torch.nn as nn
import os
import numpy as np
from einops import rearrange
from tqdm import tqdm


class TankSloshingData(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 train_mode=True,  # return an array with points' value not sampled set to zero
                 ):
        self.data_dir = args.data_dir
        self.case_len = args.case_len
        self.dataset_stat = args.dataset_stat
        self.num_case = args.num_case
        self.in_tw = args.in_tw
        self.out_tw = args.out_tw

        self.f_lst = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.f_lst.sort()

        idxs = np.arange(min(self.num_case, len(self.f_lst)))
        np.random.seed(44)  # deterministic
        np.random.shuffle(idxs)
        print(idxs[:20])

        self.train_mode = train_mode
        idxs = idxs[:]
        self.train_idxs = idxs[:int(0.9 * len(idxs))]
        if train_mode:
            print('Using training data')
            self.idxs = idxs[:int(0.9 * len(idxs))]
        else:
            print('Using testing data')
            self.idxs = idxs[int(0.9 * len(idxs)):]

        self.cache = {}
        if os.path.exists(self.dataset_stat):
            print('Loading dataset stats from', self.dataset_stat)
            stats = np.load(self.dataset_stat, allow_pickle=True)
            # npz to dict
            self.prepare_data(calculate_stats=False)
            self.stats = {k: stats[k] for k in stats.files}
        else:
            print('Calculating dataset stats')
            self.prepare_data(calculate_stats=True)
            print('Saving dataset stats to', self.dataset_stat)
            np.savez(self.dataset_stat, **self.stats, allow_pickle=True)
        print(f'Dataset stats: {self.stats}')

    def __len__(self):
        if self.train_mode:
            return len(self.idxs) * (self.case_len - self.in_tw - self.out_tw)
        else:
            return len(self.idxs)

    def prepare_data(self, calculate_stats):
        vel_total = []
        prs_total = []
        param_total = []

        # coordinates
        x0, y0 = np.meshgrid(np.linspace(0, 1, 121),
                             np.linspace(0, 0.5, 61))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 61, 121]
        self.grid = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c').float()  # [61*121, 2]

        for idx in self.idxs:
            path = self.f_lst[idx]
            path = os.path.join(self.data_dir, path)
            data = np.load(path)
            vel_all = data['vel']
            prs_all = data['prs']
            vof_all = data['vof']
            # param = data['height']

            if vel_all.shape[1] > 61:
                vel_all = vel_all[:, :61, :]
                prs_all = prs_all[:, :61, :]
                vof_all = vof_all[:, :61, :]

            vel_total.append(vel_all)
            prs_total.append(prs_all)
            # param_total.append(param)
            assert self.case_len <= vel_all.shape[0]
            self.cache[idx] = (vel_all, prs_all, vof_all)

        if calculate_stats:
            vel_all = np.concatenate(vel_total, axis=0)
            prs_all = np.concatenate(prs_total, axis=0)

            # perform normalization
            vel_mean, vel_std = np.mean(vel_all), np.std(vel_all)
            prs_mean, prs_std = np.mean(prs_all), np.std(prs_all)
            # param_min, param_max = np.min(param_total), np.max(param_total)

            print('Data prepared')

            self.stats = {'vel_mean': vel_mean,
                          'vel_std': vel_std,
                          'prs_mean': prs_mean,
                          'prs_std': prs_std,
                          # 'param_min': 0.0,   # give some offset
                          # 'param_max': 100.0,
                          'height': self.cache[list(self.cache.keys())[0]][0].shape[1],
                          'width': self.cache[list(self.cache.keys())[0]][0].shape[2]
                          }

            print(f'Sample shape: {self.cache[list(self.cache.keys())[0]][0].shape}')
            assert self.grid.shape[0] == self.cache[list(self.cache.keys())[0]][0].shape[1] * \
                   self.cache[list(self.cache.keys())[0]][0].shape[2]

    def normalize_data(self, vel, prs):
        vel = (vel - self.stats['vel_mean']) / self.stats['vel_std']
        prs = (prs - self.stats['prs_mean']) / self.stats['prs_std']
        # if param > self.stats['param_max'] or param < self.stats['param_min']:
        #     print(param, self.stats['param_max'], self.stats['param_min'])
        #     raise ValueError('Parameter out of range')
        #
        # param = (param - self.stats['param_min']) / (self.stats['param_max'] - self.stats['param_min'])

        return vel, prs         #, param

    @torch.no_grad()
    def encode_dataset(self, vq_ae, device):
        # before second stage training, we can encode the data first to save up time
        # encode the data into indices
        print('Encoding data...')
        self.encoded_data = {}
        for idx, data in tqdm(self.cache.items()):
            vel_in, prs_in, vof_in = self.cache[idx]
            vel_in, prs_in = self.normalize_data(vel_in, prs_in)
            t = vel_in.shape[0]
            u = np.concatenate((vel_in, prs_in[..., None], vof_in[..., None]), axis=-1)  # (h, w, 4)
            u = rearrange(torch.from_numpy(u), 't h w c -> t c h w').float().to(device)
            z_lst = []
            for i in range(0, t, 32):
                if i + 32 > t:
                    u_ = u[i:]
                else:
                    u_ = u[i:i + 32]
                z_lst.append(vq_ae.encode(u_))  # [t, c, h, w]
            z = torch.cat(z_lst, dim=0)
            self.encoded_data[idx] = z.detach().cpu().clone()

    def __getitem__(self, idx):
        if self.train_mode:
            case_idx = idx // self.case_len
            seed_to_read = self.idxs[case_idx]
        else:
            case_idx = idx
            seed_to_read = self.idxs[case_idx]

        vel, prs, vof = self.cache[seed_to_read]
        if self.train_mode:
            input_t = idx % (self.case_len - self.in_tw - self.out_tw)
        else:
            input_t = np.arange(self.case_len)
        # sample random snapshot
        vel_in = vel[input_t]
        prs_in = prs[input_t]
        vof_in = vof[input_t]

        # vof is already in range (0, 1), no need to normalize
        vel_in, prs_in = self.normalize_data(vel_in, prs_in)
        x = np.concatenate((vel_in, prs_in[..., None], vof_in[..., None]), axis=-1)  # (h, w, 4)
        if self.train_mode:
            z = self.encoded_data[seed_to_read][input_t:input_t+self.in_tw+self.out_tw].clone().float()
            z_in = z[:self.in_tw]
            z_out = z[self.in_tw:]
            return z_in, z_out

        else:
            x = rearrange(torch.from_numpy(x), 't h w c -> t c h w').float()
            return x[:self.in_tw], x[self.in_tw:]

    def denormalize(self,
                    x,  # [b, (t,) c, h, w]
                    ):
        # x: b h w c
        # impose Dirichlet boundary condition
        x = x.clone()
        # velocity
        x[..., :2, :, :] = x[..., :2, :, :] * self.stats['vel_std'].item() + self.stats['vel_mean'].item()
        # four closed boundaries
        x[..., :2, 0, :] = 0.
        x[..., :2, -1, :] = 0.
        x[..., :2, :, 0] = 0.
        x[..., :2, :, -1] = 0.

        # pressure
        x[..., 2, :, :] = x[..., 2, :, :] * self.stats['prs_std'].item() + self.stats['prs_mean'].item()

        # clamp vof
        x[..., 3, :, :] = torch.clamp(x[..., 3, :, :], 0., 1.+1e-8)

        return x


class ConditionalTankSloshingData(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 train_mode=True,  # return an array with points' value not sampled set to zero
                 ):
        self.data_dir = args.data_dir
        self.case_len = args.case_len
        self.dataset_stat = args.dataset_stat
        self.num_case = args.num_case
        self.in_tw = args.in_tw
        self.out_tw = args.out_tw

        self.f_lst = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.f_lst.sort()
        idxs = np.arange(min(self.num_case, len(self.f_lst)))
        np.random.seed(44)  # deterministic
        np.random.shuffle(idxs)

        self.train_mode = train_mode
        idxs = idxs[:]
        self.train_idxs = idxs[:int(0.9 * len(idxs))]
        if train_mode:
            print('Using training data')
            self.idxs = idxs[:int(0.9 * len(idxs))]
        else:
            print('Using testing data')
            self.idxs = idxs[int(0.9 * len(idxs)):]

        self.cache = {}
        if os.path.exists(self.dataset_stat):
            print('Loading dataset stats from', self.dataset_stat)
            stats = np.load(self.dataset_stat, allow_pickle=True)
            # npz to dict
            self.prepare_data(calculate_stats=False)
            self.stats = {k: stats[k] for k in stats.files}
        else:
            print('Calculating dataset stats')
            self.prepare_data(calculate_stats=True)
            print('Saving dataset stats to', self.dataset_stat)
            np.savez(self.dataset_stat, **self.stats, allow_pickle=True)
        print(f'Dataset stats: {self.stats}')

    def __len__(self):
        if self.train_mode:
            return len(self.idxs) * (self.case_len - self.in_tw - self.out_tw)
        else:
            return len(self.idxs)

    def prepare_data(self, calculate_stats):
        vel_total = []
        prs_total = []
        param_total = []

        # coordinates
        x0, y0 = np.meshgrid(np.linspace(0, 1, 121),
                             np.linspace(0, 0.5, 61))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 61, 121]
        self.grid = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c').float()  # [61*121, 2]

        for idx in self.idxs:
            path = self.f_lst[idx]
            path = os.path.join(self.data_dir, path)
            data = np.load(path)
            vel_all = data['vel']
            prs_all = data['prs']
            vof_all = data['vof']
            param = data['freq']

            if vel_all.shape[1] > 61:
                vel_all = vel_all[:, :61, :]
                prs_all = prs_all[:, :61, :]
                vof_all = vof_all[:, :61, :]

            vel_total.append(vel_all)
            prs_total.append(prs_all)
            param_total.append(param)
            assert self.case_len <= vel_all.shape[0]
            self.cache[idx] = (vel_all, prs_all, vof_all, param)

        if calculate_stats:
            vel_all = np.concatenate(vel_total, axis=0)
            prs_all = np.concatenate(prs_total, axis=0)

            # perform normalization
            vel_mean, vel_std = np.mean(vel_all), np.std(vel_all)
            prs_mean, prs_std = np.mean(prs_all), np.std(prs_all)
            param_min, param_max = np.min(param_total), np.max(param_total)

            print('Data prepared')

            self.stats = {'vel_mean': vel_mean,
                          'vel_std': vel_std,
                          'prs_mean': prs_mean,
                          'prs_std': prs_std,
                          'param_min': param_min - 2.,  # give some offset
                          'param_max': param_max + 2.,
                          'height': self.cache[list(self.cache.keys())[0]][0].shape[1],
                          'width': self.cache[list(self.cache.keys())[0]][0].shape[2]
                          }

            print(f'Sample shape: {self.cache[list(self.cache.keys())[0]][0].shape}')
            assert self.grid.shape[0] == self.cache[list(self.cache.keys())[0]][0].shape[1] * \
                   self.cache[list(self.cache.keys())[0]][0].shape[2]

    def normalize_data(self, vel, prs, param):
        vel = (vel - self.stats['vel_mean']) / self.stats['vel_std']
        prs = (prs - self.stats['prs_mean']) / self.stats['prs_std']
        if param > self.stats['param_max'] or param < self.stats['param_min']:
            print(param, self.stats['param_max'], self.stats['param_min'])
            raise ValueError('Parameter out of range')

        param = (param - self.stats['param_min']) / (self.stats['param_max'] - self.stats['param_min'])

        return vel, prs, param

    @torch.no_grad()
    def encode_dataset(self, vq_ae, device):
        # before second stage training, we can encode the data first to save up time
        # encode the data into indices
        print('Encoding data...')
        self.encoded_data = {}
        for idx, data in tqdm(self.cache.items()):
            vel_in, prs_in, vof_in, param_in = self.cache[idx]
            vel_in, prs_in, param_in = self.normalize_data(vel_in, prs_in, param_in)
            t = vel_in.shape[0]
            u = np.concatenate((vel_in, prs_in[..., None], vof_in[..., None]), axis=-1)  # (h, w, 4)
            u = rearrange(torch.from_numpy(u), 't h w c -> t c h w').float().to(device)
            z_lst = []
            for i in range(0, t, 32):
                if i + 32 > t:
                    u_ = u[i:]
                else:
                    u_ = u[i:i + 32]
                z_lst.append(vq_ae.encode(u_))  # [t, c, h, w]
            z = torch.cat(z_lst, dim=0)
            self.encoded_data[idx] = z.detach().cpu().clone()

    def __getitem__(self, idx):
        if self.train_mode:
            case_idx = idx // self.case_len
            seed_to_read = self.idxs[case_idx]
        else:
            case_idx = idx
            seed_to_read = self.idxs[case_idx]

        vel, prs, vof, param = self.cache[seed_to_read]
        if self.train_mode:
            input_t = idx % (self.case_len - self.in_tw - self.out_tw)
        else:
            input_t = np.arange(self.case_len)
        # sample random snapshot
        vel_in = vel[input_t]
        prs_in = prs[input_t]
        vof_in = vof[input_t]

        # vof is already in range (0, 1), no need to normalize
        vel_in, prs_in, param = self.normalize_data(vel_in, prs_in, param)
        x = np.concatenate((vel_in, prs_in[..., None], vof_in[..., None]), axis=-1)  # (h, w, 4)
        if self.train_mode:
            z = self.encoded_data[seed_to_read][input_t:input_t+self.in_tw+self.out_tw].clone().float()
            z_in = z[:self.in_tw]
            z_out = z[self.in_tw:]
            return z_in, z_out, param

        else:
            x = rearrange(torch.from_numpy(x), 't h w c -> t c h w').float()
            return x[:self.in_tw], x[self.in_tw:], param

    def denormalize(self,
                    x,  # [b, (t,) c, h, w]
                    ):
        # x: b h w c
        # impose Dirichlet boundary condition
        x = x.clone()
        # velocity
        x[..., :2, :, :] = x[..., :2, :, :] * self.stats['vel_std'].item() + self.stats['vel_mean'].item()
        # four closed boundaries
        x[..., :2, 0, :] = 0.
        x[..., :2, -1, :] = 0.
        x[..., :2, :, 0] = 0.
        x[..., :2, :, -1] = 0.

        # pressure
        x[..., 2, :, :] = x[..., 2, :, :] * self.stats['prs_std'].item() + self.stats['prs_mean'].item()

        # clamp vof
        x[..., 3, :, :] = torch.clamp(x[..., 3, :, :], 0., 1.+1e-8)

        return x


class SimpleTankSloshingData(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 train_mode=True,  # return an array with points' value not sampled set to zero
                 ):
        self.data_dir = args.data_dir
        self.case_len = args.case_len
        self.dataset_stat = args.dataset_stat
        self.num_case = args.num_case
        self.in_tw = args.in_tw
        self.out_tw = args.out_tw

        self.f_lst = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.f_lst.sort()
        idxs = np.arange(min(self.num_case, len(self.f_lst)))
        np.random.seed(44)  # deterministic
        np.random.shuffle(idxs)

        self.train_mode = train_mode
        idxs = idxs[:]
        self.train_idxs = idxs[:int(0.9 * len(idxs))]
        if train_mode:
            print('Using training data')
            self.idxs = idxs[:int(0.9 * len(idxs))]
        else:
            print('Using testing data')
            self.idxs = idxs[int(0.9 * len(idxs)):]

        self.cache = {}
        if os.path.exists(self.dataset_stat):
            print('Loading dataset stats from', self.dataset_stat)
            stats = np.load(self.dataset_stat, allow_pickle=True)
            # npz to dict
            self.prepare_data(calculate_stats=False)
            self.stats = {k: stats[k] for k in stats.files}
        else:
            print('Calculating dataset stats')
            self.prepare_data(calculate_stats=True)
            print('Saving dataset stats to', self.dataset_stat)
            np.savez(self.dataset_stat, **self.stats, allow_pickle=True)
        print(f'Dataset stats: {self.stats}')

    def __len__(self):
        if self.train_mode:
            return len(self.idxs) * (self.case_len - self.in_tw - self.out_tw)
        else:
            return len(self.idxs)

    def prepare_data(self, calculate_stats):
        vel_total = []
        prs_total = []
        param_total = []

        # coordinates
        x0, y0 = np.meshgrid(np.linspace(0, 1, 121),
                             np.linspace(0, 0.5, 61))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 61, 121]
        self.grid = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c').float()  # [61*121, 2]

        for idx in self.idxs:
            path = self.f_lst[idx]
            path = os.path.join(self.data_dir, path)
            data = np.load(path)
            vel_all = data['vel']
            prs_all = data['prs']
            vof_all = data['vof']
            # param = data['height']

            if vel_all.shape[1] > 61:
                vel_all = vel_all[:, :61, :]
                prs_all = prs_all[:, :61, :]
                vof_all = vof_all[:, :61, :]

            vel_total.append(vel_all)
            prs_total.append(prs_all)
            # param_total.append(param)
            assert self.case_len <= vel_all.shape[0]
            self.cache[idx] = (vel_all, prs_all, vof_all)

        if calculate_stats:
            vel_all = np.concatenate(vel_total, axis=0)
            prs_all = np.concatenate(prs_total, axis=0)

            # perform normalization
            vel_mean, vel_std = np.mean(vel_all), np.std(vel_all)
            prs_mean, prs_std = np.mean(prs_all), np.std(prs_all)
            # param_min, param_max = np.min(param_total), np.max(param_total)

            print('Data prepared')

            self.stats = {'vel_mean': vel_mean,
                          'vel_std': vel_std,
                          'prs_mean': prs_mean,
                          'prs_std': prs_std,
                          # 'param_min': 0.0,   # give some offset
                          # 'param_max': 100.0,
                          'height': self.cache[list(self.cache.keys())[0]][0].shape[1],
                          'width': self.cache[list(self.cache.keys())[0]][0].shape[2]
                          }

            print(f'Sample shape: {self.cache[list(self.cache.keys())[0]][0].shape}')
            assert self.grid.shape[0] == self.cache[list(self.cache.keys())[0]][0].shape[1] * \
                   self.cache[list(self.cache.keys())[0]][0].shape[2]

    def normalize_data(self, vel, prs):
        vel = (vel - self.stats['vel_mean']) / self.stats['vel_std']
        prs = (prs - self.stats['prs_mean']) / self.stats['prs_std']
        # if param > self.stats['param_max'] or param < self.stats['param_min']:
        #     print(param, self.stats['param_max'], self.stats['param_min'])
        #     raise ValueError('Parameter out of range')
        #
        # param = (param - self.stats['param_min']) / (self.stats['param_max'] - self.stats['param_min'])

        return vel, prs         #, param

    def __getitem__(self, idx):
        if self.train_mode:
            case_idx = idx // self.case_len
            seed_to_read = self.idxs[case_idx]
        else:
            case_idx = idx
            seed_to_read = self.idxs[case_idx]

        vel, prs, vof = self.cache[seed_to_read]
        if self.train_mode:
            input_t = idx % (self.case_len - self.in_tw - self.out_tw)
            # sample random snapshot
            vel_in = vel[input_t:input_t + self.in_tw+self.out_tw]
            prs_in = prs[input_t:input_t + self.in_tw+self.out_tw]
            vof_in = vof[input_t:input_t + self.in_tw+self.out_tw]
        else:
            input_t = np.arange(self.case_len)
            vel_in = vel[input_t]
            prs_in = prs[input_t]
            vof_in = vof[input_t]

        # vof is already in range (0, 1), no need to normalize
        vel_in, prs_in = self.normalize_data(vel_in, prs_in)
        x = np.concatenate((vel_in, prs_in[..., None], vof_in[..., None]), axis=-1)  # (h, w, 4)
        if self.train_mode:
            x = rearrange(torch.from_numpy(x), 't h w c -> t c h w').float()
            x_in = x[:self.in_tw]
            x_out = x[self.in_tw:]
            return x_in, x_out

        else:
            x = rearrange(torch.from_numpy(x), 't h w c -> t c h w').float()
            return x[:self.in_tw], x[self.in_tw:]

    def denormalize(self,
                    x,  # [b, (t,) c, h, w]
                    ):
        # x: b h w c
        # impose Dirichlet boundary condition
        # velocity
        x[..., :2, :, :] = x[..., :2, :, :].clone() * self.stats['vel_std'].item() + self.stats['vel_mean'].item()
        # four closed boundaries
        x[..., :2, 0, :] = 0.
        x[..., :2, -1, :] = 0.
        x[..., :2, :, 0] = 0.
        x[..., :2, :, -1] = 0.


        # pressure
        x[..., 2, :, :] = x[..., 2, :, :].clone() * self.stats['prs_std'].item() + self.stats['prs_mean'].item()

        # clamp vof
        x[..., 3, :, :] = torch.clamp(x[..., 3, :, :].clone(), 0., 1.+1e-8)

        return x

    def impose_dirichlet_to_normalized_input(self, x):
        # assume x is in [b c h w] and is normalized
        # impose dirichlet boundary condition
        # four closed boundaries
        x = x.clone()
        normalized_bv = (0. - self.stats['vel_mean'].item()) / self.stats['vel_std'].item()
        x[..., :2, 0, :] = normalized_bv
        x[..., :2, -1, :] = normalized_bv
        x[..., :2, :, 0] = normalized_bv
        x[..., :2, :, -1] = normalized_bv

        return x


class ConditionalSimpleTankSloshingData(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 train_mode=True,  # return an array with points' value not sampled set to zero
                 ):
        self.data_dir = args.data_dir
        self.case_len = args.case_len
        self.dataset_stat = args.dataset_stat
        self.num_case = args.num_case
        self.in_tw = args.in_tw
        self.out_tw = args.out_tw

        self.f_lst = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
        self.f_lst.sort()
        idxs = np.arange(min(self.num_case, len(self.f_lst)))
        np.random.seed(44)  # deterministic
        np.random.shuffle(idxs)

        self.train_mode = train_mode
        idxs = idxs[:]
        self.train_idxs = idxs[:int(0.9 * len(idxs))]
        if train_mode:
            print('Using training data')
            self.idxs = idxs[:int(0.9 * len(idxs))]
        else:
            print('Using testing data')
            self.idxs = idxs[int(0.9 * len(idxs)):]

        self.cache = {}
        if os.path.exists(self.dataset_stat):
            print('Loading dataset stats from', self.dataset_stat)
            stats = np.load(self.dataset_stat, allow_pickle=True)
            # npz to dict
            self.prepare_data(calculate_stats=False)
            self.stats = {k: stats[k] for k in stats.files}
        else:
            print('Calculating dataset stats')
            self.prepare_data(calculate_stats=True)
            print('Saving dataset stats to', self.dataset_stat)
            np.savez(self.dataset_stat, **self.stats, allow_pickle=True)
        print(f'Dataset stats: {self.stats}')

    def __len__(self):
        if self.train_mode:
            return len(self.idxs) * (self.case_len - self.in_tw - self.out_tw)
        else:
            return len(self.idxs)

    def prepare_data(self, calculate_stats):
        vel_total = []
        prs_total = []
        param_total = []

        # coordinates
        x0, y0 = np.meshgrid(np.linspace(0, 1, 121),
                             np.linspace(0, 0.5, 61))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 61, 121]
        self.grid = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c').float()  # [61*121, 2]

        for idx in self.idxs:
            path = self.f_lst[idx]
            path = os.path.join(self.data_dir, path)
            data = np.load(path)
            vel_all = data['vel']
            prs_all = data['prs']
            vof_all = data['vof']
            param = data['freq']

            if vel_all.shape[1] > 61:
                vel_all = vel_all[:, :61, :]
                prs_all = prs_all[:, :61, :]
                vof_all = vof_all[:, :61, :]

            vel_total.append(vel_all)
            prs_total.append(prs_all)
            param_total.append(param)
            assert self.case_len <= vel_all.shape[0]
            self.cache[idx] = (vel_all, prs_all, vof_all, param)

        if calculate_stats:
            vel_all = np.concatenate(vel_total, axis=0)
            prs_all = np.concatenate(prs_total, axis=0)

            # perform normalization
            vel_mean, vel_std = np.mean(vel_all), np.std(vel_all)
            prs_mean, prs_std = np.mean(prs_all), np.std(prs_all)
            param_min, param_max = np.min(param_total), np.max(param_total)

            print('Data prepared')

            self.stats = {'vel_mean': vel_mean,
                          'vel_std': vel_std,
                          'prs_mean': prs_mean,
                          'prs_std': prs_std,
                          'param_min': param_min - 2.,  # give some offset
                          'param_max': param_max + 2.,
                          'height': self.cache[list(self.cache.keys())[0]][0].shape[1],
                          'width': self.cache[list(self.cache.keys())[0]][0].shape[2]
                          }

            print(f'Sample shape: {self.cache[list(self.cache.keys())[0]][0].shape}')
            assert self.grid.shape[0] == self.cache[list(self.cache.keys())[0]][0].shape[1] * \
                   self.cache[list(self.cache.keys())[0]][0].shape[2]

    def normalize_data(self, vel, prs, param):
        vel = (vel - self.stats['vel_mean']) / self.stats['vel_std']
        prs = (prs - self.stats['prs_mean']) / self.stats['prs_std']
        if param > self.stats['param_max'] or param < self.stats['param_min']:
            print(param, self.stats['param_max'], self.stats['param_min'])
            raise ValueError('Parameter out of range')

        param = (param - self.stats['param_min']) / (self.stats['param_max'] - self.stats['param_min'])

        return vel, prs, param

    def __getitem__(self, idx):
        if self.train_mode:
            case_idx = idx // self.case_len
            seed_to_read = self.idxs[case_idx]
        else:
            case_idx = idx
            seed_to_read = self.idxs[case_idx]

        vel, prs, vof, param = self.cache[seed_to_read]
        if self.train_mode:
            input_t = idx % (self.case_len - self.in_tw - self.out_tw)
            # sample random snapshot
            vel_in = vel[input_t:input_t + self.in_tw+self.out_tw]
            prs_in = prs[input_t:input_t + self.in_tw+self.out_tw]
            vof_in = vof[input_t:input_t + self.in_tw+self.out_tw]
        else:
            input_t = np.arange(self.case_len)
            vel_in = vel[input_t]
            prs_in = prs[input_t]
            vof_in = vof[input_t]

        # vof is already in range (0, 1), no need to normalize
        vel_in, prs_in, param = self.normalize_data(vel_in, prs_in, param)
        x = np.concatenate((vel_in, prs_in[..., None], vof_in[..., None]), axis=-1)  # (h, w, 4)

        if self.train_mode:
            x = rearrange(torch.from_numpy(x), 't h w c -> t c h w').float()
            x_in = x[:self.in_tw]
            x_out = x[self.in_tw:]
            return x_in, x_out, param

        else:
            x = rearrange(torch.from_numpy(x), 't h w c -> t c h w').float()
            return x[:self.in_tw], x[self.in_tw:], param

    def denormalize(self,
                    x,  # [b, (t,) c, h, w]
                    ):
        # x: b h w c
        # impose Dirichlet boundary condition
        # velocity
        x[..., :2, :, :] = x[..., :2, :, :].clone() * self.stats['vel_std'].item() + self.stats['vel_mean'].item()
        # four closed boundaries
        x[..., :2, 0, :] = 0.
        x[..., :2, -1, :] = 0.
        x[..., :2, :, 0] = 0.
        x[..., :2, :, -1] = 0.


        # pressure
        x[..., 2, :, :] = x[..., 2, :, :].clone() * self.stats['prs_std'].item() + self.stats['prs_mean'].item()

        # clamp vof
        x[..., 3, :, :] = torch.clamp(x[..., 3, :, :].clone(), 0., 1.+1e-8)

        return x

    def impose_dirichlet_to_normalized_input(self, x):
        # assume x is in [b c h w] and is normalized
        # impose dirichlet boundary condition
        # four closed boundaries
        x = x.clone()
        normalized_bv = (0. - self.stats['vel_mean'].item()) / self.stats['vel_std'].item()
        x[..., :2, 0, :] = normalized_bv
        x[..., :2, -1, :] = normalized_bv
        x[..., :2, :, 0] = normalized_bv
        x[..., :2, :, -1] = normalized_bv

        return x