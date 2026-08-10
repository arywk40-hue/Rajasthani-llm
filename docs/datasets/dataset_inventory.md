# Dataset Inventory — Rajasthani Dialect AI

This document provides a comprehensive inventory of speech, text, parallel MT, and TTS datasets across the 6 major Rajasthani dialect varieties.

## Master Dialect Matrix

Codes below are ISO 639-3. Earlier revisions of this file used invented tags of the form
`raj_xxx`; those are not registered identifiers and have been removed. Note that `mwr`
(Marwari) is a *macrolanguage* covering several of these varieties — the project stays at
the individual-language level and uses `rwr` for Marwari specifically.

| Language / Dialect | Text Corpus | Speech Corpus | Parallel MT Corpus | TTS Corpus | Collected so far |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Marwari** (`rwr`) | Required | Required | Required | Required | **None** |
| **Mewari** (`mtr`) | Required | Required | Required | Required | **None** |
| **Dhundhari** (`dhd`) | Required | Required | Required | Required | **None** |
| **Hadoti** (`hoj`) | Required | Required | Required | Required | **None** |
| **Mewati** (`wtm`) | Required | Required | Required | Required | **None** |
| **Bagri** (`bgq`) | Required | Required | Required | Required | **None** |

The "Collected so far" column is not a schedule slipping — it is the current state of the
repository. `data/raw/vaani/` is empty, no audio file is present for any dialect, and the
only cached text (10,000 Karya rows) carries the single undifferentiated label
`rajasthani`. Every "Required" in this table is still outstanding.

---

## Primary Data Sources

Each entry is a source the project intends to draw on. None has been ingested yet.

1. **VAANI (ARTPARK-IISc / Google):** Spontaneous speech across 165 districts, ~31,000 hours
   total with ~2,043 hours transcribed. The corpus is publicly fetchable and is published as
   one config per district (`Rajasthan_<District>`). `src/data/fetch_datasets.py` derives
   candidate configs from `DISTRICT_DIALECT_MAP`, so all six dialect belts are attempted;
   VAANI does not necessarily cover every Rajasthan district, and unavailable configs are
   skipped with a warning. The fetch has not been run to completion here.
2. **Karya Speech Dataset (`severo/speech-rj-hi`):** 426,873 read-speech recordings from 98
   participants in Soda, Rajasthan. What is cached locally is 10,000 **text-only** rows with
   no `audio_path` field, in standard Hindi register — useful for ASR acoustics once the
   audio is fetched, but not a source of dialect text.
3. **AI4Bharat BPCC:** Parallel sentences intended for experience replay during MT
   fine-tuning. Not fetched.
4. **Custom Linguistic Sets (`data/linguistic/`):** Evaluation sets for idioms, cultural
   expressions, and code-switching. Check the directory for what is actually present before
   citing coverage.

For detailed dialect-specific metadata, consult:
- [Marwari Dataset Specification](marwari.md)
- [Mewari Dataset Specification](mewari.md)
- [Dhundhari Dataset Specification](dhundhari.md)
- [Hadoti Dataset Specification](hadoti.md)
- [Mewati Dataset Specification](mewati.md)
- [Bagri Dataset Specification](bagri.md)
