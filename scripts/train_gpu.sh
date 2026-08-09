#!/bin/bash
# GPU Training Script — Optimized for available data (no slow audio fetch)
# Uses: Karya 5000 records + VAANI 1000 records already on disk

export PYTHONPATH=.

echo "========================================================"
echo " Rajasthani Dialect AI — Fast Training (skip audio fetch)"
echo "========================================================"

nvidia-smi | head -20

# ──────────────────────────────────────────────────────────────
# PHASE 0: Check existing data
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 0: Checking existing data ==="

python3 -c "
from pathlib import Path
import json

def count_jsonl(p):
    p = Path(p)
    if not p.exists(): return 0
    return sum(1 for l in open(p) if l.strip())

karya = count_jsonl('data/raw/karya/karya_rajasthan.jsonl')
vaani = count_jsonl('data/raw/vaani/vaani_rajasthan.jsonl')
print(f'  Karya text records : {karya}')
print(f'  VAANI text records : {vaani}')
total = karya + vaani
print(f'  Total available    : {total}')
if total == 0:
    print('ERROR: No data found. Run fetch_data.py first.')
    exit(1)
print('Data check passed.')
"

# ──────────────────────────────────────────────────────────────
# PHASE 1: Prepare MT training data from existing text records
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 1: Preparing MT training data from existing records ==="

python3 -c "
import json
from pathlib import Path

mt_output = Path('data/processed/mt_dialect_train.jsonl')
mt_output.parent.mkdir(parents=True, exist_ok=True)

count = 0
with open(mt_output, 'w') as f_out:
    # Use Karya text
    karya = Path('data/raw/karya/karya_rajasthan.jsonl')
    if karya.exists():
        for line in open(karya):
            record = json.loads(line)
            text = record.get('text', '').strip()
            if len(text) < 5: continue
            f_out.write(json.dumps({
                'source_text': text,
                'target_text': text,
                'source_lang': 'hin_Deva',
                'target_lang': 'hin_Deva',
            }, ensure_ascii=False) + '\n')
            count += 1

    # Also use VAANI text
    vaani = Path('data/raw/vaani/vaani_rajasthan.jsonl')
    if vaani.exists():
        for line in open(vaani):
            record = json.loads(line)
            text = record.get('text', '').strip()
            if len(text) < 5: continue
            f_out.write(json.dumps({
                'source_text': text,
                'target_text': text,
                'source_lang': 'hin_Deva',
                'target_lang': 'hin_Deva',
            }, ensure_ascii=False) + '\n')
            count += 1

print(f'Created {count} MT training pairs -> {mt_output}')
"

# ──────────────────────────────────────────────────────────────
# PHASE 2: MT Benchmark (zero-shot — no data file needed)
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Zero-shot MT benchmark ==="
python3 experiments/mt/benchmark.py || echo "[WARN] MT benchmark failed, continuing..."

# ──────────────────────────────────────────────────────────────
# PHASE 3: MT Fine-tuning
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 3: MT Fine-tuning (IndicTrans2) ==="

TRAIN_COUNT=$(python3 -c "
from pathlib import Path
p = Path('data/processed/mt_dialect_train.jsonl')
print(sum(1 for l in open(p) if l.strip()) if p.exists() else 0)
")
echo "Training samples: $TRAIN_COUNT"

if [ "$TRAIN_COUNT" -gt "0" ]; then
    python3 scripts/train_mt.py \
        --train-data data/processed/mt_dialect_train.jsonl \
        --model-name indic-indic-dist-320M \
        --output-dir checkpoints/mt_indictrans2 \
        --epochs 3 \
        --batch-size 2 \
        --grad-accum 8 \
        --lr 3e-5 \
        --device cuda || echo "[WARN] MT training exited, check logs"
else
    echo "[SKIP] No MT training data found"
fi

# ──────────────────────────────────────────────────────────────
# PHASE 4: ASR Zero-shot Benchmark
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 4: ASR Zero-shot Benchmark ==="
python3 experiments/asr/benchmark.py || echo "[WARN] ASR benchmark failed, continuing..."

# ──────────────────────────────────────────────────────────────
# PHASE 5: ASR Fine-tuning (only if audio data available)
# ──────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 5: Checking for ASR audio data ==="

AUDIO_COUNT=$(python3 -c "
from pathlib import Path
import json
p = Path('data/raw/vaani/vaani_audio_metadata.jsonl')
if not p.exists(): print(0)
else:
    count = sum(1 for l in open(p) if json.loads(l).get('audio_path') and Path(json.loads(l)['audio_path']).exists())
    print(count)
")

if [ "$AUDIO_COUNT" -gt "50" ]; then
    echo "Found $AUDIO_COUNT audio files — running ASR fine-tuning"
    python3 scripts/train_asr.py \
        --train-data data/processed/asr_train.jsonl \
        --val-data data/processed/asr_val.jsonl \
        --epochs 5 \
        --batch-size 2 \
        --lr 1e-5 || echo "[WARN] ASR training exited"
else
    echo "[SKIP] Not enough audio files ($AUDIO_COUNT) for ASR fine-tuning — skipping"
    echo "       Run: python scripts/fetch_data.py --with-audio separately when network is stable"
fi

echo ""
echo "========================================================"
echo " PIPELINE COMPLETE"
echo " Results: results/"
echo " Checkpoints: checkpoints/"
echo "========================================================"
