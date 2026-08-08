# Data Governance & Licensing Framework

## 1. Ethical Data Governance

Data governance for low-resource Indian languages requires cultural sensitivity, fair credit to native speakers, and strict adherence to open-science principles.

### Community Representation & Co-Ownership
- Dataset curation involves linguistic verification by native Rajasthani speakers across 6 regions (Jodhpur, Udaipur, Jaipur, Kota, Alwar, Sri Ganganagar).
- Cultural expressions, regional folklore, and agricultural terms are annotated preserving local nuance without artificial standardization.

---

## 2. Licensing Inventory

| Dataset | Primary Dialects | Source | License | Permissible Commercial Use |
| :--- | :--- | :--- | :--- | :--- |
| **VAANI** | Bagri, Marwari, Mewari | ARTPARK-IISc / Google | CC-BY-4.0 | Yes (with attribution) |
| **Karya (speech-rj-hi)** | Marwari, Dhundhari | Karya / HF (severo) | Open / CC-BY-NC-4.0 | Non-Commercial / Research |
| **LDC-IL Golden Set** | Marwari | CIIL / LDC-IL | Academic License | Academic Research Only |
| **BPCC Parallel Corpus** | Rajasthani / Hindi / English | AI4Bharat | CC-BY-4.0 | Yes (with attribution) |

---

## 3. Data Lineage & Provenance Tracking

- **Raw Data (`data/raw/`):** Immutable copy of raw downloads with original metadata.
- **Processed Data (`data/processed/`):** Cleaned, normalized, and split datasets tagged with timestamp and script version.
- **Metadata Records (`data/metadata/`):** Full provenance records mapping audio files to speaker ID, district, gender, and sample rate.
