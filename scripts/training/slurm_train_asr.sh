#!/bin/bash
#SBATCH --job-name=rajasthani-asr-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_asr_%j.log
#SBATCH --error=logs/slurm_asr_%j.err

echo "=== Starting ASR Fine-Tuning Job on IIT Mandi HPC ==="
echo "Node: $(hostname)"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Load modules if required by cluster
# module load cuda/11.8 python/3.10

# Activate Virtual Environment
source venv/bin/activate || source ~/.bashrc

# Ensure log directory exists
mkdir -p logs

# Run Distributed ASR Training with PyTorch Accelerate / DDP
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    src/asr/trainer.py \
    --config config/asr.yaml \
    --model_id vasista22/whisper-hindi-large-v2 \
    --dataset_path data/raw/karya/karya_rajasthan.jsonl \
    --output_dir models/checkpoints/asr \
    --epochs 5 \
    --batch_size 16 \
    --lr 1e-4

echo "=== ASR Training Job Complete ==="
