#!/bin/bash
# Use this script to launch distributed training
# The python script needs to be launched from launch.py:
# https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md

#!/bin/bash
#SBATCH --exclusive
#SBATCH --reservation=test
#SBATCH --job-name=frnn-torchrun     # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=00:55:00          # total run time limit (HH:MM:SS)


export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

#export OMP_NUM_THREADS=4

module purge
module load anaconda3/2021.11
conda activate frnn_env

srun python train_frnn_ddc.py         \
            --train_size 0.75         \
            --num_epochs 25           \
            --lstm_hidden_size 128    \
            --batch_size 64           \
            --seq_length 128          \
            --lstm_num_layers 4       \
            --early_stopping "false"  \
            --patience 4

#torchrun train_frnn_ddc.py --standalone --nnodes 1 --nproc_per_node "4" --rdzv-backend=c10d --rdzv-endpoint=$master_addr:0
#python /home/mg6433/.conda/envs/frnn_env/lib/python3.10/site-packages/torch/distributed/launch.py --nnode=1 --node_rank=0 --nproc_per_node=4 train_frnn_ddc.py 
#python /home/mg6433/.conda/envs/frnn_env/lib/python3.10/site-packages/torch/distributed/launch.py --nnode=2 --node_rank=1 --nproc_per_node=4 train_frnn_ddc.py 