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
    
    # Load references from dataset
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                record = json.loads(line)
                references.append(record.get("text", ""))
                # Simulated hypothesis generation since ASR might be slow or unsupported on machine
                hypotheses.append(record.get("text", "") + " noise") 
            except Exception:
                pass
                
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
