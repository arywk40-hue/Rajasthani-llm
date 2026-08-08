# Marwari (मारवाड़ी) Dataset Specification

## Overview
Marwari is the most widely spoken Rajasthani variety, predominant in Western Rajasthan (Jodhpur, Bikaner, Jaisalmer, Barmer).

## Metadata Summary
- **Dataset Name:** Marwari Speech & Parallel MT Corpus
- **Source:** VAANI (ARTPARK-IISc) / Karya / LDC-IL
- **License:** CC-BY-4.0 / Academic Research
- **Language Code:** `raj_mar` (ISO 639-3: `rwr`)
- **Speech Audio Duration:** ~85 Hours
- **Sampling Rate:** 16,000 Hz / 22,050 Hz
- **Text Script:** Devanagari (with Nukta and ळ)
- **Train/Val/Test Split:** 80% / 10% / 10%

## Quality Control & Validation
- Format validation: 16kHz mono WAV / MP3
- Unicode normalization: DevanagariNormalizer applied (Nukta & ळ preserved)
- Silence/Noise threshold: SNR > 15 dB
