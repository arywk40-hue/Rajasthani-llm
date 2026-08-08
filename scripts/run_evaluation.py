"""
Run the evaluation harness against the LDC-IL GOLDEN test set (roadmap §6).

The LDC-IL Rajasthani corpus is held strictly out of training. This script:
    - loads the golden parallel corpus
    - runs a translation model over the source side
    - computes chrF++ (and optionally CER/WER for ASR transcripts)
    - emits a JSONL scorecard + prints a human-readable summary

Usage:
    # Evaluate MT
    python scripts/run_evaluation.py --task mt --golden data/processed/ldcil_golden.jsonl --model-path checkpoints/mt_indictrans2/best

    # Evaluate ASR
    python scripts/run_evaluation.py --task asr --test-data data/processed/asr_val.jsonl --model-path checkpoints/asr_whisper/final
"""

import argparse
import json
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import Evaluator
from src.mt.model import IndicTrans2MT
from src.asr.model import WhisperASR


def load_golden(path: str | Path) -> list[tuple[str, str]]:
    path = Path(path)
    pairs: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            src = record.get("source_text") or record.get("source")
            ref = record.get("target_text") or record.get("reference") or record.get("text")
            if src and ref:
                pairs.append((src, ref))
    return pairs


def load_asr_data(path: str | Path) -> tuple[list[str], list[str]]:
    path = Path(path)
    audio_paths = []
    references = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ap = record.get("audio_path")
            ref = record.get("text")
            if ap and ref and Path(ap).exists():
                audio_paths.append(ap)
                references.append(ref)
    return audio_paths, references


def main() -> None:
    parser = argparse.ArgumentParser(description="Score models against evaluation sets")
    parser.add_argument("--task", type=str, choices=["mt", "asr"], required=True)
    parser.add_argument("--test-data", type=str, default=None, help="Input test data")
    parser.add_argument("--golden", type=str, default=None, help="Golden parallel text (for MT)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint (None=pretrained base)")
    parser.add_argument("--limit", type=int, default=None, help="Only score first N pairs")
    parser.add_argument("--output", type=str, default="reports/evaluation_report.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    
    args = parser.parse_args()

    logger.info(f"Running Evaluation Pipeline | Task: {args.task.upper()}")
    evaluator = Evaluator()

    if args.task == "mt":
        if not args.golden:
            args.golden = "data/processed/ldcil_golden.jsonl"
            
        pairs = load_golden(args.golden)
        if args.limit:
            pairs = pairs[: args.limit]
            
        logger.info(f"Loaded {len(pairs)} golden pairs from {args.golden}")

        # Load model
        model = IndicTrans2MT()
        if args.model_path:
            model.load_checkpoint(args.model_path)
            
        sources = [src for src, _ in pairs]
        references = [ref for _, ref in pairs]
        hypotheses = []
        
        logger.info("Translating...")
        for i in tqdm(range(0, len(sources), args.batch_size)):
            batch = sources[i : i + args.batch_size]
            batch_hyps = model.translate(batch, src_lang="hi", tgt_lang="hi") # Use hi as generic proxy
            hypotheses.extend(batch_hyps)
            
        report = evaluator.evaluate_mt(hypotheses, references, meta=sources)
        
        print("\n=== MT Evaluation (chrF++) ===")
        print(f"  Pairs evaluated : {report.samples}")
        print(f"  Mean chrF++     : {report.mean_chrf:.4f}")

    elif args.task == "asr":
        if not args.test_data:
            args.test_data = "data/processed/asr_val.jsonl"
            
        audio_paths, references = load_asr_data(args.test_data)
        if args.limit:
            audio_paths = audio_paths[: args.limit]
            references = references[: args.limit]
            
        logger.info(f"Loaded {len(audio_paths)} audio files from {args.test_data}")

        # Load model
        model = WhisperASR()
        if args.model_path:
            model.load_checkpoint(args.model_path)
            
        logger.info("Transcribing...")
        hypotheses = model.transcribe(audio_paths, language="hi", batch_size=args.batch_size)
        
        report = evaluator.evaluate_asr(hypotheses, references)
        
        print("\n=== ASR Evaluation ===")
        print(f"  Samples evaluated : {report.samples}")
        print(f"  Mean CER          : {report.mean_cer:.4f}")
        print(f"  Mean WER          : {report.mean_wer:.4f}")

    if args.output:
        evaluator.save_report(report, args.output)

    logger.info("Evaluation Complete")


if __name__ == "__main__":
    sys.exit(main())