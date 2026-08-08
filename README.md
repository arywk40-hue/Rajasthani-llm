# Rajasthani Dialect AI

This project develops robust, production-grade Artificial Intelligence (AI) language technologies for Marwari, Mewari, Dhundhari, Hadoti, Mewati, and Bagri dialects.

It uses a cascaded Speech-to-Speech Translation (S2ST) approach:
1.  **ASR (FastConformer):** Transcribes dialect audio to Devanagari text.
2.  **MT (IndicTrans2):** Translates Devanagari text to the target language (e.g., English, Standard Hindi).
3.  **TTS (FastPitch + HiFi-GAN):** Synthesizes human-like speech from text.

## Features
- Strict Devanagari normalization pipeline (Nukta, ळ preservation).
- Cross-lingual transfer learning from standard Hindi.
- Quantized edge execution (INT8/FP16) for Suno Sutra hardware.
- Fully offline capable and DPDP-Act compliant (India-First localization).

## Setup
```bash
make setup
```
