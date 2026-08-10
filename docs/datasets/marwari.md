# Marwari (मारवाड़ी) Dataset Specification

## Overview
Marwari is the most widely spoken Rajasthani variety, predominant in Western Rajasthan (Jodhpur, Bikaner, Jaisalmer, Barmer).

## Metadata Summary
- **Dataset Name:** Marwari Speech & Parallel MT Corpus
- **Source:** VAANI (ARTPARK-IISc) / Karya / LDC-IL
- **License:** CC-BY-4.0 / Academic Research
- **Language Code:** ISO 639-3 `rwr` (Marwari, India). Note `mwr` is the *macrolanguage* covering several Rajasthani varieties — use `rwr` for this dialect specifically and stay at one level of the hierarchy across the project.
- **Speech Audio Duration — TARGET:** ~85 hours
- **Speech Audio Duration — ACTUAL:** **0 hours.** No audio file for this dialect exists in the repository.
- **Sampling Rate:** 16,000 Hz / 22,050 Hz
- **Text Script:** Devanagari (with Nukta and ळ)
- **Train/Val/Test Split (planned):** 80% / 10% / 10% — not yet created

## Collection Status

**Nothing in this document is a description of collected data.** It is a specification of
what to collect. `data/raw/vaani/` is empty; the cached Karya rows are text-only, labelled
`rajasthani` without dialect distinction, and in standard Hindi register.

## Quality Control & Validation

The checks below are the intended acceptance criteria; none have been applied, as there is
no audio to apply them to.

- Format validation: 16kHz mono WAV / MP3
- Unicode normalization: DevanagariNormalizer applied (Nukta & ळ preserved) — *this component is implemented and unit-tested*
- Silence/Noise threshold: SNR > 15 dB
