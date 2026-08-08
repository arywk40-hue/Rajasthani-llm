"""
Zero-Shot ASR Benchmark Runner

Runs IndicWhisper, Whisper-large-v3, and other baselines on VAANI test data
to establish WER/CER baselines for the hackathon (Track 1).

Outputs: reports/benchmark_<model>_<timestamp>.json

Usage:
    python scripts/run_benchmark.py --model indicwhisper-hindi-large-v2 --test-data data/processed/asr_val.jsonl
    python scripts/run_benchmark.py --model whisper-large-v3 --test-data data/processed/asr_val.jsonl --max-samples 500
    python scripts/run_benchmark.py --all-models --test-data data/processed/asr_val.jsonl
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.asr.model import WhisperASR, WHISPER_MODELS
from src.evaluation.metrics import compute_cer, compute_wer


def run_benchmark(
    model_name: str,
    test_data: Path,
    max_samples: int = 500,
    language: str = "hi",
) -> dict:
    """Run zero-shot benchmark for a single model."""
    logger.info(f"Loading model: {model_name}")
    asr = WhisperASR(model_name=model_name)
    
    # Load test data
    audio_paths = []
    references = []
    
    with open(test_data, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                record = json.loads(line.strip())
                ap = record.get("audio_path", "")
                text = record.get("text", "")
                if ap and text and Path(ap).exists():
                    audio_paths.append(ap)
                    references.append(text)
            except json.JSONDecodeError:
                continue
    
    if not audio_paths:
        logger.error("No valid test samples with existing audio files")
        return {}
    
    logger.info(f"Running benchmark on {len(audio_paths)} samples...")
    start = time.time()
    
    # Transcribe
    transcripts = asr.transcribe(audio_paths, language=language)
    
    elapsed = time.time() - start
    
    # Compute metrics
    cers = []
    wers = []
    per_sample = []
    
    for hyp, ref, ap in zip(transcripts, references, audio_paths):
        cer = compute_cer(hyp, ref)
        wer = compute_wer(hyp, ref)
        cers.append(cer)
        wers.append(wer)
        per_sample.append({
            "audio_path": ap,
            "hypothesis": hyp,
            "reference": ref,
            "cer": cer,
            "wer": wer,
        })
    
    report = {
        "model": model_name,
        "hf_model_id": WHISPER_MODELS.get(model_name, model_name),
        "test_data": str(test_data),
        "language": language,
        "samples": len(audio_paths),
        "mean_cer": sum(cers) / len(cers),
        "mean_wer": sum(wers) / len(wers),
        "std_cer": (sum((c - sum(cers)/len(cers))**2 for c in cers) / len(cers))**0.5,
        "std_wer": (sum((w - sum(wers)/len(wers))**2 for w in wers) / len(wers))**0.5,
        "time_seconds": elapsed,
        "time_per_sample": elapsed / len(audio_paths),
        "per_sample": per_sample,
        "timestamp": datetime.now().isoformat(),
    }
    
    logger.success(
        f"Benchmark complete: CER={report['mean_cer']:.4f} "
        f"WER={report['mean_wer']:.4f} ({elapsed:.1f}s total)"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description="Run zero-shot ASR benchmarks")
    parser.add_argument("--model", type=str, default=None, help="Model name from WHISPER_MODELS")
    parser.add_argument("--all-models", action="store_true", help="Run all available models")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test JSONL")
    parser.add_argument("--max-samples", type=int, default=500, help="Max test samples")
    parser.add_argument("--language", type=str, default="hi", help="Whisper language code")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    args = parser.parse_args()

    test_path = Path(args.test_data)
    if not test_path.exists():
        logger.error(f"Test data not found: {test_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all_models:
        models = list(WHISPER_MODELS.keys())
        logger.info(f"Running benchmarks for {len(models)} models...")
    else:
        model = args.model or "indicwhisper-hindi-large-v2"
        models = [model]

    all_reports = []
    for model_name in models:
        try:
            report = run_benchmark(model_name, test_path, args.max_samples, args.language)
            if report:
                all_reports.append(report)
                # Save individual report
                out_file = output_dir / f"benchmark_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved: {out_file}")
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
            continue

    # Save summary
    if all_reports:
        summary_file = output_dir / f"benchmark_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(all_reports, f, ensure_ascii=False, indent=2)
        logger.success(f"Summary saved: {summary_file}")

        # Print summary table
        print("\n" + "="*80)
        print("ZERO-SHOT ASR BENCHMARK SUMMARY")
        print("="*80)
        print(f"{'Model':<35} {'Samples':>8} {'CER':>8} {'WER':>8} {'Time/s':>8}")
        print("-"*80)
        for r in all_reports:
            print(f"{r['model']:<35} {r['samples']:>8} {r['mean_cer']:>8.4f} {r['mean_wer']:>8.4f} {r['time_per_sample']:>8.3f}")
        print("="*80)


if __name__ == "__main__":
    main()