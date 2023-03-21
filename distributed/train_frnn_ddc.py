#!/usr/bin/env python
#-*- coding: utf-8 -*-

import argparse
import os
from os.path import join 
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random
from time import perf_counter

import logging 
logging.basicConfig(filename="FRNN_distributed_test.log",
                 filemode='w',
                 format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                 datefmt='%H:%M:%S',
                 level=logging.DEBUG)


import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import sys
sys.path.append("/home/rkube/repos/d3d_loaders")


from d3d_loaders.time_sampling import sampler_causal, sampler_linearip, sampler_space
from d3d_loaders.standardizers import standardizer_mean_std
from d3d_loaders.d3d_loaders import Multishot_dataset
from d3d_loaders.samplers import BatchedSampler_multi_dist,  collate_fn_batched



def run(local_world_size, local_rank):
    rank_str = f"rank {local_rank}/{local_world_size}"
    n_gpus = torch.cuda.device_count() // local_world_size

    logging.info("{rank_str} - using {n_gpu} gpus")

    device_ids = list(range(local_rank * n_gpus, (local_rank + 1) * n_gpus))
    logging.info(f"[{os.getpid()}] {rank_str}:  n_gpus = {n_gpus}, device_ids = {device_ids}"
    )

    # Model parameters
    seq_length = 128    # Sequence length used for prediction
    batch_size = 64     # Number of sequences per batch
    lstm_hidden_size = 128  # Dimension of LSTM hidden state
    lstm_num_layers = 4     # Number of LSTM layers

    # Dataset parameters
    num_train = 10 # Number of shots used for training
    num_valid = 2  # Number of shots used for validation

    # fetch dataset definition for list of predictors.
    datapath = "/projects/FRNN/dataset_D3D_100/D3D_100"
    with open(join(datapath, "..", "d3d_100.yaml"), "r") as fp:
        d3d_100 = yaml.safe_load(fp)


    with open(join(datapath, "..", "shots_t_min_max.yaml"), "r") as fp:
        t_min_max_dict = yaml.safe_load(fp)

    # The d3d_datasets expects a that the samplers for each shot are passed as a dictionary.
    sampler_pred_dict = {}
    sampler_targ_dict = {}

    # Instantiate the samplers and store in the dict.
    for shotnr in d3d_100["shots"].keys():
        tmin = t_min_max_dict[shotnr]["tmin"]
        tmax = t_min_max_dict[shotnr]["tmax"]
        sampler_pred_dict.update({shotnr: sampler_causal(tmin, tmax, 1.0, t_shift=0.0)})
        sampler_targ_dict.update({shotnr: sampler_linearip(tmin, tmax, 1.0, t_shift=0.0)})

    # Instantiate signal normalizers
    norm_dict = {}
    with open(join(datapath, "..", "normalization.yaml"), "r") as fp:
        normalization = yaml.safe_load(fp)

    for k, v in normalization.items():
        norm_dict[k] = standardizer_mean_std(v["mean"], v["std"])

    # Instantiate a sampler for the profiles.
    # This will re-sample edensfit and etempfit on psi=[0.0, 1.0] with 32 points

    ip_profile = sampler_space(np.linspace(0.0, 1.0, 32, dtype=np.float32))

    # To a random test/train split over the shots.
    shot_list = list(d3d_100["shots"].keys())
    random.seed(1337)
    random.shuffle(shot_list)

    # Split
    num_shots = len(shot_list)


    shots_train = shot_list[:num_train]
    shots_valid = shot_list[num_train:num_train + num_valid]

    # Instantiate datasets
    # Create the training set. This can take some time.
    ds_train = Multishot_dataset(shots_train, d3d_100["predictors"], ["ttd"],
                                 sampler_pred_dict, sampler_targ_dict, ip_profile, norm_dict, datapath, torch.device("cpu"))
    shot_length_train = []
    for ix, shotnr in enumerate(shots_train):
        shot_length_train.append(ds_train.shot(ix).__len__())
        logging.info(f"{rank_str} - Training set: shot {shotnr} - {shot_length_train[-1]} samples")

    # Create the validation set and print stats on length
    ds_valid = Multishot_dataset(shots_valid, d3d_100["predictors"], ["ttd"],
                                 sampler_pred_dict, sampler_targ_dict, ip_profile, norm_dict, datapath, torch.device("cpu"))
    shot_length_valid = []
    for ix, shotnr in enumerate(shots_valid):
        shot_length_valid.append(ds_valid.shot(ix).__len__())
        logging.info(f"{rank_str} - Validation set: shot {shotnr} - {shot_length_valid[-1]} samples")


    # Instantiate a loader for training and validation sets.
    # This loader picks 'batch_size' sequences of size seq_length+1 at random starting points, distributed across shots.
    # Use shuffle=True to shuffle the dataset. Note that the sampler sets a seed for shuffling. During training
    # we have to manually set the seed so that the shuffling will be different in every epoch
    sampler_train = BatchedSampler_multi_dist(shot_length_train, seq_length, batch_size, shuffle=True)

    loader_train = DataLoader(ds_train,
                            batch_sampler=sampler_train, 
                            collate_fn=collate_fn_batched())

    # Get performance metrics over iterations
    t_start = perf_counter()
    cnt = 0
    for x, y in loader_train:
        cnt = cnt + 1
    t_end = perf_counter()
    logging.info(f"{rank_str} iteration over training set: {cnt} samples, takes {(t_end - t_start):7.3f}s")
    

    sampler_valid = BatchedSampler_multi_dist(shot_length_valid, seq_length, batch_size, shuffle=True)

    loader_valid = DataLoader(ds_valid,
                            batch_sampler=sampler_valid,
                            collate_fn=collate_fn_batched())

    t_start = perf_counter()
    cnt = 0
    for x, y in loader_valid:
        cnt = cnt + 1
    t_end = perf_counter()
    logging.info(f"{rank_str} iteration over validation set: {cnt} samples, takes {(t_end - t_start):7.3f}s")


    ##### Build a simple LSTM model
    class simple_lstm(nn.Module):
        def __init__(self, num_classes, input_size, hidden_size, seq_length, num_layers=2):
            super(simple_lstm, self).__init__()
            self.num_classes = num_classes  # Number of output features
            self.input_size = input_size    # Number of features in the input x
            self.hidden_size = hidden_size  # Number of features in hidden state h
            self.seq_length = seq_length  
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(input_size=self.input_size, 
                                hidden_size=self.hidden_size,
                                num_layers=self.num_layers, 
                                batch_first=False) #lstm
            self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
            self.fc_2 = nn.Linear(128, num_classes) #fully connected last layer

            self.relu = nn.ReLU()
        
        def forward(self, x):
            h_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size)).to(device_ids[0]) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size)).to(device_ids[0]) #internal state
            # Propagate input through LSTM
            output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
            
            hn = hn[-1, ...]    # Take hidden state of the last recurrent layer and use it for prediction
            out = F.relu(hn)
            out = self.fc_1(out) #first Dense
            out = F.relu(out) #relu
            out = self.fc_2(out)  # Final Output
            return out

    model = simple_lstm(1, len(d3d_100["predictors"]), hidden_size=lstm_hidden_size, seq_length=seq_length, num_layers=lstm_num_layers).to(device_ids[0])
    ddp_model = DDP(model, device_ids)

    loss_fn = nn.MSEloss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=5e-4)


def spmd_main(rank, size, backend="nccl"):
    print("Hello, world")

    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    run(rank, size)
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local_world_size", type=int, default=1)
    args = parser.parse_args()

    spmd_main(args.local_world_size, args.local_rank)



# end of file train_frnn_ddc.py
