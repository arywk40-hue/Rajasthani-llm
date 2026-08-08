# Privacy Policy & Data Governance

## Overview

The **Rajasthani Dialect AI** platform is committed to respecting user privacy, protecting speech/text data, and ensuring compliance with applicable data protection laws, including India's **Digital Personal Data Protection (DPDP) Act, 2023**.

---

## 1. Data Collection & Principles

- **Minimal Data Collection:** We collect only audio recordings and transcriptions voluntarily provided for model inference or dataset contribution.
- **Explicit Consent:** Audio collection during field work or interactive app usage requires informed consent from speakers or local community representatives.
- **Anonymization:** Personally Identifiable Information (PII) such as phone numbers, government IDs, and specific personal addresses are scrubbed during data validation (`scripts/data/validate_text.py`).

---

## 2. Audio & Text Data Storage

- **Inference Mode:** Audio streams processed through our FastAPI backend (`src/api/app.py`) are held ephemerally in RAM during ASR/TTS execution and discarded immediately after processing unless explicitly opted into research logging.
- **Dataset Storage:** Fine-tuning datasets (e.g., VAANI, Karya, LDC-IL splits) are stored in restricted cloud storage or local disk (`data/raw/`, `data/processed/`) with strict role-based access.

---

## 3. Local vs. Cloud Execution

- **Cloud API Fallbacks:** When using Bhashini or HuggingFace endpoints for TTS/ASR, requests are transmitted via TLS 1.3 encryption.
- **Local Edge Inference:** Quantized ONNX/PyTorch INT8 models run entirely on-device, ensuring zero audio transmission to external servers.

---

## 4. Rights of Data Principals

Data contributors have the right to:
1. Request access to their contributed audio samples.
2. Request deletion of their voice recordings from future dataset releases.
3. Opt out of research logging at any time.
