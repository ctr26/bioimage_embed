#!/bin/bash -l
#SBATCH --nodes=3
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-24:00:00
#SBATCH --job-name=lightning
#SBATCH --constraint=a100  # Ensure the job is scheduled on nodes with A100 GPUs
#SBATCH --mem-per-cpu=2GB
#SBATCH --cpus-per-task=32
#SBATCH --output=lightning_%j.out
set -x

source activate $1

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
# export NCCL_P2P_DISABLE=1
# unset LOCAL_RANK

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
echo "Starting Lightning training script"
srun python3 scripts/idr/study.py