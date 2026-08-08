#!/bin/bash
#SBATCH --job-name=rajasthani-mt-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_mt_%j.log
#SBATCH --error=logs/slurm_mt_%j.err

echo "=== Starting MT Fine-Tuning Job (IndicTrans2 + Model Souping) on IIT Mandi HPC ==="
echo "Node: $(hostname)"
echo "CUDA Available Devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi

source venv/bin/activate || source ~/.bashrc

mkdir -p logs

# Run Distributed MT Training
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    src/mt/trainer.py \
    --config config/mt.yaml \
    --model_id AI4Bharat/IndicTrans2-en-indic-1B \
    --dataset_path data/raw/karya/karya_rajasthan.jsonl \
    --output_dir models/checkpoints/mt \
    --epochs 10 \
    --batch_size 32 \
    --lr 5e-5

echo "=== MT Training Job Complete ==="
