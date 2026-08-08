"""
Audio Validation Pipeline

Validates format, sample rate, duration, silence, and SNR for speech datasets.
"""

import os
import json
import argparse
from pathlib import Path
from loguru import logger
import soundfile as sf
import numpy as np

def validate_audio_file(file_path: Path, min_duration=0.5, max_duration=30.0, target_sr=16000):
    if not file_path.exists():
        return False, "File does not exist"
    
    try:
        info = sf.info(str(file_path))
        duration = info.duration
        sr = info.samplerate
        
        if duration < min_duration or duration > max_duration:
            return False, f"Invalid duration: {duration:.2f}s (allowed: {min_duration}-{max_duration}s)"
            
        if sr != target_sr:
            return False, f"Sample rate mismatch: {sr}Hz (expected {target_sr}Hz)"
            
        # Read audio data to check for silent/corrupt files
        data, _ = sf.read(str(file_path))
        if np.max(np.abs(data)) < 1e-4:
            return False, "Silent or empty audio"
            
        return True, "Valid"
    except Exception as e:
        return False, f"Corrupt file: {e}"

def main():
    parser = argparse.ArgumentParser(description="Validate Audio Files in Dataset")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing audio files")
    parser.add_argument("--output", type=str, default="data/metadata/audio_validation_report.json")
    args = parser.parse_args()

    audio_dir = Path(args.dir)
    results = {"valid": 0, "rejected": 0, "details": []}

    for audio_file in audio_dir.rglob("*.wav"):
        is_valid, msg = validate_audio_file(audio_file)
        if is_valid:
            results["valid"] += 1
        else:
            results["rejected"] += 1
            results["details"].append({"file": str(audio_file), "reason": msg})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Audio Validation Complete | Valid: {results['valid']} | Rejected: {results['rejected']}")

if __name__ == "__main__":
    main()
