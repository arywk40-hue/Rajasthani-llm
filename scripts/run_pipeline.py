#!/usr/bin/env python3
"""
End-to-End Speech-to-Speech Translation Pipeline CLI

Executes the full cascaded S2ST pipeline on an input audio file.
"""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.pipeline import SpeechToSpeechPipeline


def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Speech-to-Speech translation")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (WAV)")
    parser.add_argument("--dialect", type=str, required=True, 
                        help="Input dialect (marwari, mewari, dhundhari, hadoti, mewati, bagri)")
    parser.add_argument("--target-lang", type=str, default="hindi", 
                        help="Target translation language (hindi, english, default: hindi)")
    parser.add_argument("--output", type=str, default="results/e2e/", help="Output directory (default: results/e2e/)")
    
    # Model overrides
    parser.add_argument("--asr-model", type=str, default="whisper-tiny", help="ASR model name")
    parser.add_argument("--mt-model", type=str, default="indic-indic-dist-320M", help="MT model name")
    parser.add_argument("--tts-model", type=str, default="facebook/mms-tts-hin", help="TTS model name")
    parser.add_argument("--device", type=str, default="auto", help="Execution device")
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

    asr_model = args.asr_model
    if args.asr_model == "whisper-tiny" and "asr" in config_data:
        asr_model = config_data["asr"].get("model_name", args.asr_model)

    mt_model = args.mt_model
    if args.mt_model == "indic-indic-dist-320M" and "mt" in config_data:
        mt_model = config_data["mt"].get("model_name", args.mt_model)

    tts_model = args.tts_model
    if args.tts_model == "facebook/mms-tts-hin" and "tts" in config_data:
        tts_model = config_data["tts"].get("model_name", args.tts_model)

    device = args.device
    if args.device == "auto" and "pipeline" in config_data:
        device = config_data["pipeline"].get("device", args.device)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(json.dumps({"error": f"Input audio file not found: {args.audio}"}))
        sys.exit(1)

    try:
        pipeline = SpeechToSpeechPipeline(
            asr_model_name=asr_model,
            mt_model_name=mt_model,
            tts_model_name=tts_model,
            device=device,
        )
        
        result = pipeline.process(
            audio_path=audio_path,
            dialect=args.dialect,
            target_lang=args.target_lang,
            output_dir=args.output,
        )
        
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(json.dumps({"error": f"Pipeline execution failed: {str(e)}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()
