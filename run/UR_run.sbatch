#!/bin/bash
#SBATCH --job-name=amsr2_sr
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=48:00:00

echo "============================================"
echo "AMSR2 Super-Resolution Training Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 32GB per GPU"
echo "============================================"

# Set environment variables
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install required packages
echo "Installing required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user torch torchvision numpy matplotlib scikit-learn tqdm psutil

# Test environment
echo "Testing environment:"
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

echo "============================================"
echo "Starting AMSR2 SR training:"

# Change to project directory
cd $HOME/temperature_sr_project

# Create directories if they don't exist
mkdir -p ./models
mkdir -p ./logs
mkdir -p ./results

# Run training with optimized parameters for 32GB Turing GPU
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $HOME/temperature_sr_project:$HOME/temperature_sr_project \
    --env PYTHONPATH=$HOME/temperature_sr_project:$PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python amsr2_super_resolution_complete.py \
    --npz-dir /home/vdidur/temperature_sr_project/data \
    --batch-size 2 \
    --epochs 2 \
    --lr 1e-4 \
    --scale-factor 10 \
    --save-path ./models/best_amsr2_model.pth \
    --val-split 0.2

echo "============================================"
echo "AMSR2 SR Training Finished: $(date)"
echo "============================================"