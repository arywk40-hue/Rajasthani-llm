import os
import json
import argparse
from pathlib import Path
from loguru import logger
from src.evaluation.metrics import Evaluator

def main():
    parser = argparse.ArgumentParser(description="Run Zero-Shot WER Benchmark on Rajasthani Data")
    parser.add_argument("--dataset", type=str, default="data/raw/vaani/vaani_rajasthan.jsonl", help="Dataset path to benchmark on")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file")
    args = parser.parse_args()

    logger.info(f"Starting ASR Benchmark | dataset={args.dataset}")
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset {args.dataset} not found. Please run fetch_data.py first.")
        return
        
    evaluator = Evaluator()
    hypotheses = []
    references = []
    
    # Load references and audio paths from dataset
    audio_paths = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = json.loads(line)
                text = record.get("text", "")
                audio_path = record.get("audio_path", "")
                if text and audio_path and os.path.exists(audio_path):
                    references.append(text)
                    audio_paths.append(audio_path)
            except Exception:
                pass

    if not audio_paths:
        logger.error(
            "No records with valid audio_path found in dataset. "
            "Run scripts/fetch_data.py --with-audio first to download audio files."
        )
        return

    # Run real ASR inference
    logger.info(f"Running ASR inference on {len(audio_paths)} audio files...")
    try:
        from src.asr.model import WhisperASR
        asr = WhisperASR()
        hypotheses = asr.transcribe(audio_paths, language="hi")
    except Exception as e:
        logger.error(f"ASR inference failed: {e}")
        return
                
    if not references:
        logger.warning(f"No records found in {args.dataset}. Cannot compute metrics.")
        return
        
    logger.info(f"Evaluating {len(references)} records...")
    report = evaluator.evaluate_asr(hypotheses, references)
    
    results = {
        "dataset": args.dataset,
        "metrics": {
            "average_wer": report.mean_wer,
            "average_cer": report.mean_cer,
        },
        "samples_evaluated": report.samples
    }
    
    out_path = Path(args.output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    logger.success(f"Benchmark complete. Results saved to {out_path.absolute()}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
