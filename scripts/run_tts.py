#!/usr/bin/env python3
"""
TTS CLI Entry Point

Synthesizes speech from text and saves to a WAV file.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tts.synthesize import IndicTTSSynthesizer


def main():
    parser = argparse.ArgumentParser(description="Run Meta MMS VITS TTS on text")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="results/example.wav", help="Output path for WAV file")
    parser.add_argument("--model", type=str, default="facebook/mms-tts-hin", 
                        help="TTS model name or HuggingFace ID (default: facebook/mms-tts-hin)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu, cuda, mps, auto)")
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
    if args.model == "facebook/mms-tts-hin" and "tts" in config_data:
        model_name = config_data["tts"].get("model_name", args.model)

    device = args.device
    if args.device == "auto" and "pipeline" in config_data:
        device = config_data["pipeline"].get("device", args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tts = IndicTTSSynthesizer(model_name=model_name, device=device)
        tts.synthesize_to_file(args.text, output_path)
        
        # Verify generated audio file is valid
        import soundfile as sf
        info = sf.info(str(output_path))
        
        result = {
            "text": args.text,
            "output_path": str(output_path),
            "sample_rate": info.samplerate,
            "duration_sec": round(info.duration, 2),
            "measured": "true"
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": f"TTS execution failed: {str(e)}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
