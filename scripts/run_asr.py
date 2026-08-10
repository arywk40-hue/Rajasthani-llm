#!/usr/bin/env python3
"""
ASR CLI Entry Point

Transcribes a single audio file and outputs structured JSON.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.asr.inference import ASRInference


def main():
    parser = argparse.ArgumentParser(description="Run Whisper ASR on a single audio file")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (WAV)")
    parser.add_argument("--model", type=str, default="whisper-tiny", 
                        help="ASR model name or HuggingFace ID (e.g. whisper-tiny, indicwhisper-hindi-large-v2)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu, cuda, mps, auto)")
    parser.add_argument("--language", type=str, default="hi", help="Forced decode language (default: hi)")
    parser.add_argument("--config", type=str, default=None, help="Path to default.yaml configuration file")
    args = parser.parse_args()

    # Load from config if provided
    config_data = {}
    if args.config:
        try:
            import yaml
            with open(args.config, "r") as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            print(json.dumps({"error": f"Failed to load config: {e}"}))
            sys.exit(1)

    model_name = args.model
    if args.model == "whisper-tiny" and "asr" in config_data:
        model_name = config_data["asr"].get("model_name", args.model)

    device = args.device
    if args.device == "auto" and "pipeline" in config_data:
        device = config_data["pipeline"].get("device", args.device)

    language = args.language
    if args.language == "hi" and "asr" in config_data:
        language = config_data["asr"].get("language", args.language)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(json.dumps({"error": f"Audio file not found: {args.audio}"}))
        sys.exit(1)

    try:
        asr = ASRInference(model_name=model_name, device=device)
        result = asr.transcribe(str(audio_path), language=language)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": f"ASR execution failed: {str(e)}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
