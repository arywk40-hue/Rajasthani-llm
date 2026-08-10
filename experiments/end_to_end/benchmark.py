"""
End-to-End Pipeline Benchmarking Suite — Real Inference

Executes the full S2ST pipeline over the fetched/demo audio manifest,
timing each stage, verifying outputs, and calculating latency/success rates.
"""

import csv
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
from loguru import logger

from src.pipeline.pipeline import SpeechToSpeechPipeline

DIALECTS = ["marwari", "mewari", "dhundhari", "hadoti", "mewati", "bagri"]


def load_manifest(path: Path) -> list[dict]:
    """Read the dataset manifest JSONL, keeping only rows that actually exist on disk."""
    if not path.exists():
        logger.error(f"DATASET_NOT_AVAILABLE: Manifest not found at {path}")
        return []

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            audio_path = rec.get("audio_path", "")
            if audio_path and Path(audio_path).exists():
                records.append(rec)
    return records


def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Pipeline Benchmark")
    parser.add_argument("--manifest", type=str, default="data/raw/vaani/vaani_audio_metadata.jsonl")
    parser.add_argument("--output_csv", type=str, default="results/end_to_end_results.csv")
    parser.add_argument("--asr-model", type=str, default="whisper-tiny")
    parser.add_argument("--mt-model", type=str, default="indic-indic-dist-320M")
    parser.add_argument("--tts-model", type=str, default="facebook/mms-tts-hin")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit", type=int, default=1, help="Samples per dialect to evaluate (default: 1 for speed)")
    args = parser.parse_args()

    records = load_manifest(Path(args.manifest))
    if not records:
        logger.error("DATASET_NOT_AVAILABLE: No usable audio records found. Run scripts/fetch_data.py --demo first.")
        return 1

    # Group by dialect
    by_dialect = defaultdict(list)
    for rec in records:
        by_dialect[rec["dialect"]].append(rec)

    # Apply limit per dialect
    eval_records = []
    for dialect in DIALECTS:
        dialect_recs = by_dialect.get(dialect, [])
        eval_records.extend(dialect_recs[:args.limit])

    if not eval_records:
        logger.error("No valid dialect samples to evaluate.")
        return 1

    logger.info(f"Loaded {len(eval_records)} samples for S2ST End-to-End evaluation")

    try:
        pipeline = SpeechToSpeechPipeline(
            asr_model_name=args.asr_model,
            mt_model_name=args.mt_model,
            tts_model_name=args.tts_model,
            device=args.device
        )
    except Exception as e:
        logger.error(f"MODEL_NOT_AVAILABLE: Pipeline initialization failed: {e}")
        return 1

    results = []
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    output_audio_dir = out_csv.parent / "e2e_benchmark_audio"
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total_count = 0

    for idx, rec in enumerate(eval_records):
        dialect = rec["dialect"]
        audio_path = rec["audio_path"]
        logger.info(f"[{idx+1}/{len(eval_records)}] Evaluating {dialect} sample: {audio_path}")
        
        total_count += 1
        dialect_out_dir = output_audio_dir / dialect
        dialect_out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            res = pipeline.process(
                audio_path=audio_path,
                dialect=dialect,
                target_lang="hindi",
                output_dir=dialect_out_dir
            )
            success_count += 1
            results.append({
                "use_case": f"{dialect.capitalize()}_Scenario",
                "asr_status": "PASS" if res.transcript else "FAIL",
                "mt_status": "PASS" if res.translation else "FAIL",
                "tts_status": "PASS" if res.output_duration_sec > 0 else "FAIL",
                "e2e_status": "PASS",
                "avg_latency_ms": res.total_latency_ms,
                "asr_latency_ms": res.asr_latency_ms,
                "mt_latency_ms": res.mt_latency_ms,
                "tts_latency_ms": res.tts_latency_ms,
                "input_duration_sec": res.input_duration_sec,
                "output_duration_sec": res.output_duration_sec,
                "measured": "true"
            })
        except Exception as e:
            logger.error(f"E2E Pipeline execution failed for {audio_path}: {e}")
            results.append({
                "use_case": f"{dialect.capitalize()}_Scenario",
                "asr_status": "FAIL",
                "mt_status": "FAIL",
                "tts_status": "FAIL",
                "e2e_status": "FAIL",
                "avg_latency_ms": 0.0,
                "asr_latency_ms": 0.0,
                "mt_latency_ms": 0.0,
                "tts_latency_ms": 0.0,
                "input_duration_sec": 0.0,
                "output_duration_sec": 0.0,
                "measured": "true"
            })

    # Save E2E benchmark outputs
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "use_case", "asr_status", "mt_status", "tts_status", "e2e_status",
            "avg_latency_ms", "asr_latency_ms", "mt_latency_ms", "tts_latency_ms",
            "input_duration_sec", "output_duration_sec", "measured"
        ])
        writer.writeheader()
        writer.writerows(results)

    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    logger.success(f"E2E Benchmark Complete. Success rate: {success_rate:.1f}% ({success_count}/{total_count}). Results saved to {out_csv}")
    return 0 if success_rate > 0 else 1


if __name__ == "__main__":
    main()
