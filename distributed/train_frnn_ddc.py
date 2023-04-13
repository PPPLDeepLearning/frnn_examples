#!/usr/bin/env python
#-*- coding: utf-8 -*-
from socket import gethostname
#import argparse
import os
from os.path import join 
import numpy as np
import matplotlib.pyplot as plt
import yaml
import random
from time import perf_counter
from argparse import ArgumentParser
from multiprocessing import Pool

import logging 
logging.basicConfig(filename="FRNN_distributed_test.log",
                 filemode='w',
                 format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                 datefmt='%H:%M:%S',
                 level=logging.DEBUG)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import sys
sys.path.append("/scratch/gpfs/mg6433/FRNN/d3d_loaders")


from d3d_loaders.time_sampling import sampler_causal, sampler_linearip, sampler_space
from d3d_loaders.standardizers import standardizer_mean_std
from d3d_loaders.d3d_loaders import Multishot_dataset
from d3d_loaders.samplers import BatchedSampler_multi_dist,  collate_fn_batched

"""
This script demonstrates how to use the (d3d_loaders)[https://github.com/PlasmaControl/d3d_loaders/]
package for multi-gpu training of disruption prediction using pytorch distributed.

This is basically a parallel version of notebooks/FRNN_test.ipynb
"""

def get_args_parser() -> ArgumentParser:
    """Helper function parsing the command line options."""

    parser = ArgumentParser(description="FRNN Distributed Training Launcher")

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs.  Default is 10.",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="The size of training set.  Default is 0.8.",
    )
    # parser.add_argument(
    #     "--num_train",
    #     type=int,
    #     default=100,
    #     help="Number of shots used for training. Default is 100.",
    # )
    # parser.add_argument(
    #     "--num_valid",
    #     type=int,
    #     default=20,
    #     help="Number of shots used for validation. Default is 20.",
    # )

    parser.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=128,
        help="Dimension of LSTM hidden state.  Default is 128.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of sequences per batch. Default is 64.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=128,
        help="Sequence length used for prediction. Default is 128.",
    )
    parser.add_argument(
        "--lstm_num_layers",
        type=int,
        default=4,
        help="Number of LSTM layers. Default is 4.",
    )
    parser.add_argument(
        "--early_stopping",
        type=str,
        default="false",
        help="Flag for early stopping of the training. Default is false.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default="3",
        help="Number of epochs with no improvement after which training will be stopped. Default is 3.",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default="0",
        help="Minimum change to qualify as an improvement for early stopping. Default is 0.",
    )

    return parser


def parse_args():
    parser = get_args_parser()
    return parser.parse_args()

def config_from_args(args):
    config={}
    #config["num_train"]=args.num_train
    #config["num_valid"]=args.num_valid
    config["train_size"]=args.train_size
    config["num_epochs"]=args.num_epochs
    config["lstm_hidden_size"]=args.lstm_hidden_size
    config["batch_size"]=args.batch_size
    config["seq_length"]=args.seq_length
    config["lstm_num_layers"]=args.lstm_num_layers
    config["early_stopping"]=args.early_stopping
    config["min_delta"]=args.min_delta
    config["patience"]=args.patience
    return config

def run(config):

    rank = config["rank"]
    n_gpus = config["n_gpus"]
    local_world_size = config["world_size"]
    n_gpus = n_gpus // local_world_size

    local_rank = config["local_rank"]
    torch.cuda.set_device(local_rank)

    rank_str = f"rank {local_rank}/{local_world_size}"
    logging.info(f"{rank_str} - using {n_gpus} gpus")

    device_ids = list(range(local_rank * n_gpus, (local_rank + 1) * n_gpus))
    logging.info(f"[{os.getpid()}] {rank_str}:  n_gpus = {n_gpus}, device_ids = {device_ids}")

    # Model parameters
    seq_length = config["seq_length"]
    batch_size = config["batch_size"]
    lstm_hidden_size = config["lstm_hidden_size"]
    lstm_num_layers = config["lstm_num_layers"]

    # Dataset parameters
    #num_train = config["num_train"]  #100 # Number of total shots used for training. These will be split across ranks
    #num_valid = config["num_valid"]  #20 # Number of total shots used for validation. These will be split across ranks.

    if rank == 0: print(f"fetch dataset definition for list of predictors...", flush=True)
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

    if rank == 0: print(f"Instantiate signal normalizers", flush=True)
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
    train_size = config["train_size"]
    
    num_train = int(num_shots * train_size)
    num_valid = num_shots - num_train

    if rank == 0: print(f"Total available shots {num_shots}. Number of shots used for training is {num_train}", flush=True)

    shots_train_world = shot_list[:num_train]
    shots_valid_world = shot_list[num_train:num_train + num_valid]
    # Each rank has to sub-sample training and validation shots
    shots_train_rank = shots_train_world[local_rank:num_train:local_world_size]
    shots_valid_rank = shots_valid_world[local_rank:num_train:local_world_size]
    if rank == 0: print(f"{rank_str} - shots_train: {shots_train_rank}, shots_valid: {shots_valid_rank}", flush=True)

    if rank == 0: print(f"Instantiate datasets...", flush=True) 
    # Create the training set. This can take some time.
    ds_train = Multishot_dataset(shots_train_rank, d3d_100["predictors"], ["ttd"],
                                sampler_pred_dict, sampler_targ_dict, ip_profile, norm_dict, datapath, torch.device("cpu"))
  
    shot_length_train = []
    for ix, shotnr in enumerate(shots_train_rank):
        shot_length_train.append(ds_train.shot(ix).__len__())
        logging.info(f"{rank_str} - Training set: shot {shotnr} - {shot_length_train[-1]} samples")

    # Create the validation set and print stats on length
    ds_valid = Multishot_dataset(shots_valid_rank, d3d_100["predictors"], ["ttd"],
                                sampler_pred_dict, sampler_targ_dict, ip_profile, norm_dict, datapath, torch.device("cpu"))
    
    shot_length_valid = []
    for ix, shotnr in enumerate(shots_valid_rank):
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
            h_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size)).to(local_rank) #hidden state
            c_0 = Variable(torch.zeros(self.num_layers, x.size(1), self.hidden_size)).to(local_rank) #internal state
            # Propagate input through LSTM
            output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
            
            hn = hn[-1, ...]      # Take hidden state of the last recurrent layer and use it for prediction
            out = F.relu(hn)
            out = self.fc_1(out)  #first Dense
            out = F.relu(out)     #relu
            out = self.fc_2(out)  # Final Output
            return out


    class EarlyStopper:
        def __init__(self, patience=1, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.min_validation_loss = np.inf

        def early_stop(self, validation_loss):
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter = 0
            elif validation_loss > (self.min_validation_loss + self.min_delta):
                self.counter += 1
                if self.counter >= self.patience:
                    return True
            return False

    # The 76 comes from having 14 predictors:
    # 12 scalars - 12
    # 2 profiles a 32 channels - 64
    # 12 + 64 = 76
    model = simple_lstm(1, 76, hidden_size=lstm_hidden_size, seq_length=seq_length, num_layers=lstm_num_layers).to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank],output_device=local_rank)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=5e-4)

    if (config["early_stopping"] == "true"): 
        early_stopping = True
        patience = config["patience"]
        min_delta = config["min_delta"]
    else:
        early_stopping = False

    if (early_stopping): early_stopper = EarlyStopper(patience=patience)

    # Train the model. This uses about 15-40% GPU on a single A100, depending on the size of the model.

    num_epochs = config["num_epochs"]

    losses_train = np.zeros(num_epochs)
    losses_valid = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        # Remember to update the epoch of the samplers to use a new seed for sample shuffling
        sampler_train.set_epoch(epoch)
        sampler_valid.set_epoch(epoch)

        t_start = perf_counter()
        model.train()

        loss_train = 0.0
        loss_valid = 0.0
        ix_bt = 0
        for inputs, target in loader_train:
            inputs = inputs.to(local_rank) # cuda(non_blocking=True) # to(device)
            target = target.to(local_rank) # cuda(non_blocking=True) # to(device)
            optimizer.zero_grad()

            output = model(inputs[:-1, :, :])

            loss = loss_fn(output, target[-1,:,:])
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            #print(f"batch {ix_b}: loss = {loss.item()}")
            ix_bt += 1
        
        ix_bv = 0
        with torch.no_grad():
            for inputs, target in loader_valid:
                inputs = inputs.to(local_rank) # to(device)
                target = target.to(local_rank) # to(device)
                
                output = model(inputs[:-1, :, :])
            
                loss_valid += loss_fn(output, target[-1, :, :]).item()
                ix_bv += 1
                
        if (early_stopping):
            if early_stopper.early_stop(loss_valid): 
                print(f"Early stopping at epoch {epoch} for rank {local_rank}", flush=True)            
                break
                
        losses_train[epoch] = loss_train / ix_bt / batch_size
        losses_valid[epoch] = loss_valid / ix_bv / batch_size
        t_end = perf_counter()
        t_epoch = t_end - t_start

        print(f"{rank_str} : Epoch {epoch} took {t_epoch:7.3f}s. train loss = {losses_train[epoch]:8.6e}, valid loss =  {losses_valid[epoch]:8.6e}", flush=True)
    
    torch.save(model.state_dict(), f"frnn_model_{local_rank}.pt")
    
    if rank == 0: print(f"Congratulations! Training has been completed.")

def process_group_setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
            

def spmd_main():

    rank          = int(os.environ["SLURM_PROCID"])
    world_size    = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    n_gpus = gpus_per_node * int(os.environ["SLURM_NNODES"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"rank {rank} of {world_size} on {gethostname()} where there are" \
        f" {gpus_per_node} allocated GPUs per node.", flush=True)

    #dist.init_process_group("nccl", rank=rank, world_size=world_size)
    process_group_setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)

    args = parse_args()
    config = config_from_args(args)

    config["local_rank"] = local_rank
    config["rank"] = rank
    config["world_size"] = world_size
    config["rank"] = rank
    config["n_gpus"] = n_gpus

    run(config)
    dist.destroy_process_group()

if __name__ == "__main__":

    t_start = perf_counter()

    spmd_main()

    t_end = perf_counter()

    print(f"Total time is {t_end - t_start} s")



# end of file train_frnn_ddc.py