# Mewari (मेवाड़ी) Dataset Specification

## Overview
Mewari is spoken in the Mewar region of Southern Rajasthan (Udaipur, Chittorgarh, Rajsamand, Bhilwara).

## Metadata Summary
- **Dataset Name:** Mewari Spontaneous Speech & Text Corpus
- **Source:** VAANI / Field Collections
- **License:** CC-BY-4.0
- **Language Code:** ISO 639-3 `mtr`
- **Speech Audio Duration — TARGET:** ~35 hours
- **Speech Audio Duration — ACTUAL:** **0 hours.** No audio file for this dialect exists in the repository.
- **Sampling Rate:** 16,000 Hz
- **Text Script:** Devanagari
- **Train/Val/Test Split (planned):** 80% / 10% / 10% — not yet created

## Collection Status

**Nothing in this document is a description of collected data.** It is a specification of
what to collect. `data/raw/vaani/` is empty and no field collection has taken place.

## Quality Control & Validation

The checks below are the intended acceptance criteria; none have been applied, as there is
no audio to apply them to.

- Format validation: 16kHz mono WAV
- Text normalization: Standard Devanagari script verification
- Speaker distribution: Balanced gender and age coverage across Udaipur/Chittorgarh districts
