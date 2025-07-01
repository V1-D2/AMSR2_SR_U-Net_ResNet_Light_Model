#!/bin/bash
#SBATCH --job-name=amsr2_gpu_seq
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=48:00:00

echo "============================================"
echo "AMSR2 GPU Sequential Training Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 32GB per GPU"
echo "============================================"

# Set environment variables
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Change to project directory
cd /home/vdidur/AMSR2_SR_U-Net_ResNet_Light_Model

# Install required packages if needed
echo "Checking/Installing required packages..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user --quiet torch torchvision numpy matplotlib scikit-learn tqdm psutil

# Test environment
echo "Testing GPU environment..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# Create necessary directories
mkdir -p ./models
mkdir -p ./logs
mkdir -p ./results

echo "============================================"
echo "Starting GPU Sequential Training..."

# Run the GPU sequential trainer
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/AMSR2_SR_U-Net_ResNet_Light_Model:/workspace \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python gpu_sequential_trainer.py \
    --data-dir /data \
    --max-files 50 \
    --epochs-per-file 2 \
    --batch-size 4 \
    --lr 1e-4 \
    --scale-factor 10 \
    --use-amp \
    --save-path ./models/best_amsr2_gpu_model.pth

echo "============================================"
echo "Training Finished: $(date)"
echo "============================================"