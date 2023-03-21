#!/bin/bash
# Use this script to launch distributed training
# The python script needs to be launched from launch.py:
# https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md

python frnn/lib/python3.9/site-packages/torch/distributed/launch.py --nnode=1 --node_rank=0 --nproc_per_node=2 train_frnn_ddc.py --local_world_size=2
