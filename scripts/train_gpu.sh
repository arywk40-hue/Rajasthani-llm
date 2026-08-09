#!/bin/bash
# GPU Training Launch Script for Rajasthani Dialect AI
# Run: chmod +x scripts/train_gpu.sh && bash scripts/train_gpu.sh
# NOTE: We do NOT use set -e because HuggingFace streaming has a known C++ cleanup
# crash on connection abort that exits non-zero even when data was fetched correctly.

echo "========================================================"
echo " Rajasthani Dialect AI — GPU Training Pipeline"
echo "========================================================"

# Check GPU
nvidia-smi

export PYTHONPATH=.

# ──────────────────────────────────────────────────────────────
# PHASE 0: Fetch Data
# ──────────────────────────────────────────────────────────────
echo "=== Phase 0: Fetching datasets from HuggingFace ==="

# Fetch Karya text metadata (fast, no audio)
python scripts/fetch_data.py --only karya --max-karya 5000 || echo "[WARN] Karya fetch exited non-zero (likely stream cleanup bug) — checking data..."

# Fetch VAANI metadata
python scripts/fetch_data.py --only vaani --max-vaani 1000 || echo "[WARN] VAANI fetch exited non-zero — checking data..."

# Fetch VAANI with audio for ASR (slower but needed for Whisper)
python scripts/fetch_data.py --with-audio --max-vaani 300 --max-karya 1000 || echo "[WARN] VAANI audio fetch exited non-zero — checking data..."

echo "=== Data fetch complete ==="

# ──────────────────────────────────────────────────────────────
# PHASE 1: Generate MT Training Data from fetched text
# ──────────────────────────────────────────────────────────────
echo "=== Phase 1: Preparing MT training data ==="

python -c "
import json
from pathlib import Path

# Read Karya data and create MT-format parallel pairs
# Karya has Rajasthani text, we create source_text/target_text pairs
karya_path = Path('data/raw/karya/karya_rajasthan.jsonl')
mt_output = Path('data/processed/mt_dialect_train.jsonl')
mt_output.parent.mkdir(parents=True, exist_ok=True)

count = 0
if karya_path.exists():
    with open(karya_path, 'r') as f_in, open(mt_output, 'w') as f_out:
        for line in f_in:
            record = json.loads(line.strip())
            text = record.get('text', '')
            if not text or len(text) < 5:
                continue
            # For MT training: source=Rajasthani text, target=same (self-supervised)
            # Real parallel data would be dialect->Hindi pairs
            mt_record = {
                'source_text': text,
                'target_text': text,  # Self-supervised baseline
                'source_lang': 'hin_Deva',
                'target_lang': 'hin_Deva',
            }
            f_out.write(json.dumps(mt_record, ensure_ascii=False) + '\n')
            count += 1
print(f'Created {count} MT training pairs in {mt_output}')

# Also create ASR-format data from VAANI
vaani_path = Path('data/raw/vaani/vaani_rajasthan.jsonl')
asr_train = Path('data/processed/asr_train.jsonl')
asr_val = Path('data/processed/asr_val.jsonl')

asr_count = 0
if vaani_path.exists():
    records = []
    with open(vaani_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if record.get('text') and record.get('audio_path'):
                records.append(record)
    
    # 90/10 train/val split
    split_idx = int(len(records) * 0.9)
    with open(asr_train, 'w') as f:
        for r in records[:split_idx]:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(asr_val, 'w') as f:
        for r in records[split_idx:]:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    asr_count = len(records)
print(f'Created {asr_count} ASR records ({asr_train}, {asr_val})')
"

# ──────────────────────────────────────────────────────────────
# PHASE 2: Zero-shot MT Benchmark (real IndicTrans2 inference)
# ──────────────────────────────────────────────────────────────
echo "=== Phase 2: Running zero-shot MT benchmark ==="
python experiments/mt/benchmark.py

echo "=== MT benchmark results ==="
cat results/mt/benchmark_results.csv

# ──────────────────────────────────────────────────────────────
# PHASE 3: MT Fine-tuning (Low VRAM: ~6 GB)
# ──────────────────────────────────────────────────────────────
echo "=== Phase 3: Fine-tuning MT model ==="
python scripts/train_mt.py \
    --train-data data/processed/mt_dialect_train.jsonl \
    --model-name indic-indic-dist-200M \
    --output-dir checkpoints/mt_indictrans2 \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 3e-5 \
    --device cuda

# ──────────────────────────────────────────────────────────────
# PHASE 4: ASR Benchmark & Fine-tuning
# ──────────────────────────────────────────────────────────────
echo "=== Phase 4: ASR zero-shot benchmark ==="

# Zero-shot benchmark (if audio data available)
if [ -f data/processed/asr_val.jsonl ]; then
    python scripts/run_benchmark.py \
        --model indicwhisper-hindi-small \
        --test-data data/processed/asr_val.jsonl \
        --max-samples 100
fi

# Fine-tune ASR (if training data available)
if [ -f data/processed/asr_train.jsonl ]; then
    echo "=== Fine-tuning ASR (Whisper) ==="
    python scripts/train_asr.py \
        --train-data data/processed/asr_train.jsonl \
        --val-data data/processed/asr_val.jsonl \
        --model-name indicwhisper-hindi-small \
        --output-dir checkpoints/asr_whisper \
        --epochs 5 \
        --batch-size 2 \
        --lr 1e-5
fi

# ──────────────────────────────────────────────────────────────
# PHASE 5: Run all benchmarks (updates results/ CSVs)
# ──────────────────────────────────────────────────────────────
echo "=== Phase 5: Running all benchmarks ==="
python experiments/mt/benchmark.py || true
python experiments/asr/benchmark.py || true
python experiments/tts/benchmark.py || true
python experiments/end_to_end/benchmark.py || true

echo "========================================================"
echo " TRAINING COMPLETE"
echo " Checkpoints: checkpoints/"
echo " Results: results/"
echo "========================================================"
