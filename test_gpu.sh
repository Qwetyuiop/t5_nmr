#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

module load cuda/12.6

source ~/miniforge3/etc/profile.d/conda.sh
conda activate t5

echo "===== Hostname ====="
hostname

echo "===== CUDA_VISIBLE_DEVICES ====="
echo $CUDA_VISIBLE_DEVICES

echo "===== nvidia-smi ====="
nvidia-smi

echo "===== Torch check ====="
python -c "import torch; print('torch version:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"