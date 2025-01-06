import xarray as xr
import torch
from einops import rearrange
from tqdm import tqdm
import os
import numpy as np


class SW2DData(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 train_mode=True,
                 load_all=True  # load all data into memory if possible, currently it is very slow
                 ):

        self.case_len = args.case_len
        self.dataset_stat = args.dataset_stat
        self.num_case = args.num_case
        if train_mode:
            print("Loading training data")
            self.data_dir = args.train_data_dir
        else:
            print("Loading test data")
            self.data_dir = args.test_data_dir
        data = xr.open_zarr(self.data_dir)
        print(f"Dataset size: {data['u'].shape}")
        self.ndata = data['u'].shape[0]

        self.train_mode = train_mode

        self.normstat = torch.load(self.dataset_stat)

        # TODO: hard coded for now
        self.in_tw = 1
        self.interval = 2
        self.start_frame = 2              # skip the first frame

        self.out_tw = args.out_tw

        self.load_all = load_all
        if load_all:
            print("Loading all data into memory")
            self.data = {
            'u': data["u"][:self.num_case].to_numpy().astype(np.float32),
            'v': data["v"][:self.num_case].to_numpy().astype(np.float32),
            'pres': data["pres"][:self.num_case].to_numpy().astype(np.float32),
            }
            print("Done")


    def __len__(self):
        if self.train_mode:
            if (self.in_tw + self.out_tw) * self.interval + self.start_frame == self.case_len:
                return self.num_case
            else:
                return self.num_case * (self.case_len - ((self.in_tw + self.out_tw) * self.interval + self.start_frame))
        else:
            return self.ndata

    def denormalize(self, x):
        if len(x.shape) == 5:
            is_sequence = True
            t = x.shape[1]
            x = rearrange(x, 'b t c h w -> (b t) c h w')
        # assuming follow channel order: u, v, pres
        x[:, 0] = (x[:, 0] * self.normstat["u"]["std"].to(x.device)) + self.normstat["u"]["mean"].to(x.device)
        x[:, 1] = (x[:, 1] * self.normstat["v"]["std"].to(x.device)) + self.normstat["v"]["mean"].to(x.device)
        x[:, 2] = (x[:, 2] * self.normstat["pres"]["std"].to(x.device)) + self.normstat["pres"]["mean"].to(x.device)

        if is_sequence:
            x = rearrange(x, '(b t) c h w -> b t c h w', t=t)
        return x

    def normalize(self, u, v, pres):
        u = (u - self.normstat["u"]["mean"]) / self.normstat["u"]["std"]
        v = (v - self.normstat["v"]["mean"]) / self.normstat["v"]["std"]
        pres = (pres - self.normstat["pres"]["mean"]) / self.normstat["pres"]["std"]
        return u, v, pres
    
    def encode_dataset(self, vq_ae, device):
        # before second stage training, we can encode the data first to save up time
        # encode the data into indices
        print('Encoding data...')
        if not self.load_all:
            data = xr.open_zarr(self.data_dir)

        self.encoded_data = []
        for idx in tqdm(range(self.num_case)):
            if not self.load_all:
                u = torch.tensor(data["u"][idx].to_numpy().astype(np.float32))
                v = torch.tensor(data["v"][idx].to_numpy().astype(np.float32))
                pres = torch.tensor(data["pres"][idx].to_numpy().astype(np.float32))
            else:
                u = torch.tensor(self.data["u"][idx])
                v = torch.tensor(self.data["v"][idx])
                pres = torch.tensor(self.data["pres"][idx])
            u, v, pres = self.normalize(u, v, pres)

            u = u.float().to(device)
            v = v.cpu().float().to(device)
            pres = rearrange(pres.unsqueeze(-1), 't h w c -> t c h w').float().to(device)
            x_in = torch.cat((u, v, pres), dim=1)
            z = vq_ae.encode(x_in)  # [t, c, h, w]
            self.encoded_data.append(z.cpu().numpy())
        self.data = None
    
    def __getitem__(self, idx):
        if self.load_all:  # only use load all for training
            data = self.data
        else:
            data = xr.open_zarr(self.data_dir)

        if self.train_mode:
            if (self.in_tw + self.out_tw) * self.interval + self.start_frame == self.case_len:
                seed_to_read = idx
            else:
                seed_to_read = idx // (self.case_len - ((self.in_tw + self.out_tw) * self.interval + self.start_frame))
        else:
            seed_to_read = idx

        if self.train_mode:
            if (self.in_tw + self.out_tw) * self.interval + self.start_frame == self.case_len:
                start_t = self.start_frame
            else:
                start_t = self.start_frame + idx % ((self.in_tw + self.out_tw) * self.interval + self.start_frame)
            z = self.encoded_data[seed_to_read]  # [t, h*w]
            z_in = z[start_t:start_t + self.in_tw*self.interval:self.interval]  # [tin, h*w]
            z_out = z[start_t + self.in_tw*self.interval:start_t + (self.in_tw + self.out_tw)*self.interval:self.interval]  # [tout, h*w]
            return z_in, z_out
        else:
            start_t = self.start_frame
            if self.load_all:
                u = torch.tensor(data["u"][seed_to_read])
                v = torch.tensor(data["v"][seed_to_read])
                pres = torch.tensor(data["pres"][seed_to_read])
            else:
                u = torch.tensor(data["u"][seed_to_read].to_numpy())
                v = torch.tensor(data["v"][seed_to_read].to_numpy())
                pres = torch.tensor(data["pres"][seed_to_read].to_numpy())

            u = u[start_t::self.interval]
            v = v[start_t::self.interval]
            pres = pres[start_t::self.interval]
            u, v, pres = self.normalize(u, v, pres)

            x = torch.cat((u, v, pres.unsqueeze(1)), dim=1)
            x_in = x[:self.in_tw:]  # [tin, c, h, w]
            x_out = x[self.in_tw:]  # [tout, c, h, w]
            return x_in, x_out


class SW2DDataSimple(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 train_mode=True,
                 load_all=False):

        self.case_len = args.case_len
        self.dataset_stat = args.dataset_stat
        self.num_case = args.num_case
        if train_mode:
            print("Loading training data")
            self.data_dir = args.train_data_dir
        else:
            print("Loading test data")
            self.data_dir = args.test_data_dir
        data = xr.open_zarr(self.data_dir)
        print(f"Dataset size: {data['u'].shape}")
        self.data_size = data['u'].shape[0]

        self.train_mode = train_mode

        self.normstat = torch.load(self.dataset_stat)

        # TODO: hard coded for now
        self.in_tw = 1
        self.interval = 2
        self.start_frame = 2  # skip the first frame

        self.out_tw = args.out_tw

        self.load_all = load_all
        if load_all:
            print("Loading all data into memory")
            self.data = {
            'u': data["u"][:self.num_case].to_numpy().astype(np.float32),
            'v': data["v"][:self.num_case].to_numpy().astype(np.float32),
            'pres': data["pres"][:self.num_case].to_numpy().astype(np.float32),
            }
            print("Done")

    def __len__(self):
        if self.train_mode:
            if (self.in_tw + self.out_tw) * self.interval + self.start_frame == self.case_len:
                return self.num_case
            else:
                return self.num_case * (self.case_len - ((self.in_tw + self.out_tw) * self.interval + self.start_frame))
        else:
            return self.data_size

    def denormalize(self, x):
        if len(x.shape) == 5:
            is_sequence = True
            t = x.shape[1]
            x = rearrange(x, 'b t c h w -> (b t) c h w')
        # assuming follow channel order: u, v, pres
        x[:, 0] = (x[:, 0] * self.normstat["u"]["std"].to(x.device)) + self.normstat["u"]["mean"].to(x.device)
        x[:, 1] = (x[:, 1] * self.normstat["v"]["std"].to(x.device)) + self.normstat["v"]["mean"].to(x.device)
        x[:, 2] = (x[:, 2] * self.normstat["pres"]["std"].to(x.device)) + self.normstat["pres"]["mean"].to(x.device)

        if is_sequence:
            x = rearrange(x, '(b t) c h w -> b t c h w', t=t)
        return x

    def normalize(self, u, v, pres):
        u = (u - self.normstat["u"]["mean"]) / self.normstat["u"]["std"]
        v = (v - self.normstat["v"]["mean"]) / self.normstat["v"]["std"]
        pres = (pres - self.normstat["pres"]["mean"]) / self.normstat["pres"]["std"]
        return u, v, pres

    def __getitem__(self, idx):
        if self.train_mode:
            if (self.in_tw + self.out_tw) * self.interval + self.start_frame == self.case_len:
                seed_to_read = idx
            else:
                seed_to_read = idx // (self.case_len - ((self.in_tw + self.out_tw) * self.interval + self.start_frame))
        else:
            seed_to_read = idx

        if self.train_mode:
            if (self.in_tw + self.out_tw) * self.interval + self.start_frame == self.case_len:
                start_t = self.start_frame
            else:
                start_t = self.start_frame + idx % ((self.in_tw + self.out_tw) * self.interval + self.start_frame)

            if self.load_all:
                u = torch.tensor(self.data["u"][seed_to_read])
                v = torch.tensor(self.data["v"][seed_to_read])
                pres = torch.tensor(self.data["pres"][seed_to_read])

            else:
                data = xr.open_zarr(self.data_dir)
                u = torch.tensor(data["u"][seed_to_read].to_numpy())
                v = torch.tensor(data["v"][seed_to_read].to_numpy())
                pres = torch.tensor(data["pres"][seed_to_read].to_numpy())

            u = u[start_t::self.interval]
            v = v[start_t::self.interval]
            pres = pres[start_t::self.interval]
            u, v, pres = self.normalize(u, v, pres)

            x = torch.cat((u, v, pres.unsqueeze(1)), dim=1)
            x_in = x[:self.in_tw:]  # [tin, c, h, w]
            x_out = x[self.in_tw:self.in_tw+self.out_tw]  # [tout, c, h, w]
            return x_in, x_out
        else:
            start_t = self.start_frame
            if self.load_all:
                u = torch.tensor(self.data["u"][seed_to_read])
                v = torch.tensor(self.data["v"][seed_to_read])
                pres = torch.tensor(self.data["pres"][seed_to_read])
            else:
                data = xr.open_zarr(self.data_dir)
                u = torch.tensor(data["u"][seed_to_read].to_numpy())
                v = torch.tensor(data["v"][seed_to_read].to_numpy())
                pres = torch.tensor(data["pres"][seed_to_read].to_numpy())
            u = u[start_t::self.interval]
            v = v[start_t::self.interval]
            pres = pres[start_t::self.interval]
            u, v, pres = self.normalize(u, v, pres)

            x = torch.cat((u, v, pres.unsqueeze(1)), dim=1)
            x_in = x[:self.in_tw:]  # [tin, c, h, w]
            x_out = x[self.in_tw:]  # [tout, c, h, w]
            return x_in, x_out


