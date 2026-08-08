# Dataset Inventory — Rajasthani Dialect AI

This document provides a comprehensive inventory of speech, text, parallel MT, and TTS datasets across the 6 major Rajasthani dialect varieties.

## Master Dialect Matrix

| Language / Dialect | Text Corpus | Speech Corpus | Parallel MT Corpus | TTS Corpus | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Marwari** (`raj_mar`) | Required | Required | Required | Required | Validated |
| **Mewari** (`raj_mew`) | Required | Required | Required | Required | In Progress |
| **Dhundhari** (`raj_dhu`) | Required | Required | Required | Required | Validated |
| **Hadoti** (`raj_had`) | Required | Required | Required | Required | In Progress |
| **Mewati** (`raj_mwt`) | Required | Required | Required | Required | Planned |
| **Bagri** (`bgq`) | Required | Required | Required | Required | Validated |

---

## Primary Data Sources

1. **VAANI (ARTPARK-IISc / Google):** Spontaneous speech across 165 districts covering Bagri and Marwari.
2. **Karya Speech Dataset (`severo/speech-rj-hi`):** 426,000+ audio clips of Rajasthani Hindi read speech.
3. **AI4Bharat BPCC:** Parallel sentences for Rajasthani $\leftrightarrow$ Hindi / English machine translation.
4. **Custom Linguistic Sets (`data/linguistic/`):** Dedicated evaluation datasets for idioms, cultural expressions, and code-switching.

For detailed dialect-specific metadata, consult:
- [Marwari Dataset Specification](marwari.md)
- [Mewari Dataset Specification](mewari.md)
- [Dhundhari Dataset Specification](dhundhari.md)
- [Hadoti Dataset Specification](hadoti.md)
- [Mewati Dataset Specification](mewati.md)
- [Bagri Dataset Specification](bagri.md)
