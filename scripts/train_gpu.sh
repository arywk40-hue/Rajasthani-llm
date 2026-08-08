#!/bin/bash
# Convenient GPU Training Launch Script for Server Nodes (e.g. NVIDIA RTX PRO 5000)

echo "========================================================"
echo " Starting Rajasthani Dialect AI Fine-Tuning on GPU Node "
echo "========================================================"

# Check GPU status
nvidia-smi

# Set Python Path
export PYTHONPATH=.

echo "=== Detected co-existing GPU job. Configuring Low VRAM Footprint (~6 GB VRAM) ==="

# Step 1: Run MT Fine-Tuning with batch_size=2, grad_accum=8
echo "=== Phase 1: Training Machine Translation (IndicTrans2) in Low VRAM mode ==="
python src/mt/trainer.py \
    --train_data data/raw/karya/karya_rajasthan.jsonl \
    --output_dir models/checkpoints/mt_gpu \
    --epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --fp16 True

# Step 2: Run ASR Baseline & Fine-Tuning
echo "=== Phase 2: Running ASR Baseline & Fine-Tuning (Whisper) ==="
python experiments/asr/benchmark.py

echo "========================================================"
echo " Training & Benchmarking Complete! Check models/checkpoints "
echo "========================================================"
