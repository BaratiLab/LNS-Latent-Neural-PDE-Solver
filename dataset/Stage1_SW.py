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
                 load_all=False,  # load all data into memory if possible, currently it is very slow
                 ):

        self.case_len = args.case_len
        self.dataset_stat = args.dataset_stat
        self.num_case = args.num_case   # only for training
        self.load_all = load_all
        if train_mode:
            print("Loading training data")
            self.data_dir = args.train_data_dir
        else:
            print("Loading test data")
            self.data_dir = args.test_data_dir

        data = xr.open_zarr(self.data_dir)
        print(f"Dataset size: {data['u'].shape}")
        self.data_size = data['u'].shape[0]

        if not train_mode:
            self.num_case = self.data_size  # use all for testing

        self.train_mode = train_mode

        self.normstat = torch.load(self.dataset_stat)

        # TODO: hard coded for now
        self.start_frame = 2  # skip the first frame

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
            return self.num_case * \
                   (self.case_len - self.start_frame)
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
    
    def __getitem__(self, idx):
        if self.load_all:   # only use load all for training
            data = self.data
        else:
            data = xr.open_zarr(self.data_dir)

        if self.train_mode:
            seed_to_read = idx // (self.case_len - self.start_frame)
            input_t = idx % (self.case_len - self.start_frame) + self.start_frame
            if self.load_all:
                u = torch.tensor(data["u"][seed_to_read][input_t])
                v = torch.tensor(data["v"][seed_to_read][input_t])
                pres = torch.tensor(data["pres"][seed_to_read][input_t])
            else:
                u = torch.tensor(data["u"][seed_to_read][input_t].to_numpy())
                v = torch.tensor(data["v"][seed_to_read][input_t].to_numpy())
                pres = torch.tensor(data["pres"][seed_to_read][input_t].to_numpy())

            u = (u - self.normstat["u"]["mean"]) / self.normstat["u"]["std"]
            v = (v - self.normstat["v"]["mean"]) / self.normstat["v"]["std"]
            pres = (pres - self.normstat["pres"]["mean"]) / self.normstat["pres"]["std"]
            pres = pres.unsqueeze(0)
            x_in = torch.cat((u, v, pres), dim=0)
            return x_in
        else:
            if self.load_all:
                u = torch.tensor(data["u"][idx][self.start_frame:])
                v = torch.tensor(data["v"][idx][self.start_frame:])
                pres = torch.tensor(data["pres"][idx][self.start_frame:])
            else:
                u = torch.tensor(data["u"][idx][self.start_frame:].to_numpy())
                v = torch.tensor(data["v"][idx][self.start_frame:].to_numpy())
                pres = torch.tensor(data["pres"][idx][self.start_frame:].to_numpy())

            u = (u - self.normstat["u"]["mean"]) / self.normstat["u"]["std"]
            v = (v - self.normstat["v"]["mean"]) / self.normstat["v"]["std"]
            pres = (pres - self.normstat["pres"]["mean"]) / self.normstat["pres"]["std"]
            pres = pres.unsqueeze(1)
            x_all = torch.cat((u, v, pres), dim=1)
            return x_all
