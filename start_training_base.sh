#!/bin/bash

#SBATCH --output=logs/%x_%j.out
#SBATCH --job-name=t5_train_base_10000
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

echo "===== Slurm Job Info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

echo "===== Environment Before Activation ====="
echo "before activation: $(which python)"

#gpu_veryshort appears to no longer exist
#hostname
#previously the queue was gpu_veryshort
#account CHEM014742
#module load languages/miniconda
#module load libs/cuda/12.0.0-gcc-9.1.0
module load cuda/12.6

source /home/b5an/jucloud.b5an/miniforge3/bin/activate #change thsi to point to your conda installation
conda activate t5 # activate your envirnmnet

echo "===== Environment After Activation ====="
echo "after activation: $(which python)"
python --version

echo "===== GPU Info ====="
nvidia-smi

echo "===== Start Training ====="
python t5_base.py

echo "===== Job Finished ====="
echo "End time: $(date)"