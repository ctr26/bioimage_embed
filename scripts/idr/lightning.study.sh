#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-02:00:00
#SBATCH --job-name=lightning
#SBATCH --constraint=a100  # Ensure the job is scheduled on nodes with A100 GPUs
#SBATCH --mem-per-cpu=2GB
#SBATCH --cpus-per-task=32
#SBATCH --output=lightning_%j.out
set -x

source ./env/bin/activate

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# run script from above
echo "Starting Lightning training script"
srun python3 -u scripts/idr/study.py


# # shellcheck disable=SC2206
# # SBATCH --job-name=lightning
# # SBATCH --nodes=4             # This needs to match Trainer(num_nodes=...)
# # SBATCH --ntasks-per-node=1   # This needs to match Trainer(devices=...)
# # SBATCH --cpus-per-task=16
# # SBATCH --mem-per-cpu=2GB
# # SBATCH --tasks-per-node=1
# # SBATCH --gpus-per-task=1
# # SBATCH --constraint=a100  # Ensure the job is scheduled on nodes with A100 GPUs
# # SBATCH --output=lightning_%j.out
# # SBATCH --time=24:00:00