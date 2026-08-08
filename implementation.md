# Rajasthani Dialect AI — Implementation Plan & Roadmap

**Scope:** Marwari, Mewari, Dhundhari, Hadoti, Mewati, Bagri
**Target integration:** Bhashini National Hub for Language Technology (NHLT) + Suno Sutra edge device

---

## 1. Executive Summary

Build a cascaded ASR → MT → TTS pipeline for six under-resourced Rajasthani dialects, trained via cross-lingual transfer from Hindi/Gujarati and fine-tuned on a normalized, augmented fusion of existing Indic datasets. Deploy to cloud (Bhashini NHLT) and offline edge (Suno Sutra, INT8/FP16 quantized) within an 8-month build cycle, with DPDP Act-compliant data handling throughout.

**Core architecture decision:** Cascaded (ASR→MT→TTS) over end-to-end S2ST, because no parallel speech-to-translated-speech corpus exists for these dialects. Trade-off accepted: error propagation risk, mitigated by strong normalization and cross-lingual pre-training.

---

## 2. Objectives → Deliverables Map

| Objective | Deliverable |
|---|---|
| Benchmark existing text-to-text MT models (dialect ↔ English/regional) | Baseline eval report on IndicTrans2 zero-shot performance per dialect |
| Collect/validate voice + text data across all 6 dialects | Consolidated "Rajasthani-Dialect Acoustic Corpus" + validated text corpus |
| Build/train ASR, TTS, MT models | FastConformer ASR, IndicTrans2 MT, FastPitch+HiFi-GAN TTS checkpoints |
| Preserve linguistic nuance, idiom, tone, grammar | Dialect-aware normalization + phonological mapping pipeline |
| Evaluate prototypes (WER, MOS, use-case coverage) | Evaluation harness + scorecards per milestone |
| Cross-functional collaboration (researchers, linguists, devs) | Annotation/validation workflow with linguist sign-off gates |
| High accuracy, real-time feedback, seamless UX | Low-latency streaming ASR (TDT-CTC), quantized edge inference |
| DPDP Act / Bhashini privacy compliance | Encrypted pipeline, 30-day log retention, localized processing |

---

## 3. Solution Architecture

**Pipeline:** Audio/Text in → Devanagari normalization → ASR (dialect speech → Devanagari text) → MT (Devanagari → target language) → TTS (target text → speech out)

| Component | Model | Key detail |
|---|---|---|
| ASR | FastConformer + hybrid TDT/CTC decoder | Mirrors SraVaani-1.0 (31,255 hrs pretrain); 7–16 hrs dialect fine-tune outperforms Hindi-only zero-shot |
| MT | IndicTrans2 (1B param variant) | Script-unified Transformer; SentencePiece BPE for inflection; experience replay + model souping to avoid catastrophic forgetting |
| TTS | FastPitch (acoustic) + HiFi-GAN V1 (vocoder) | Outperforms end-to-end VITS on MOS for Indo-Aryan languages; explicit pitch/duration control for dialect prosody |
| Cloud deployment | Bhashini NHLT REST/WebSocket APIs | Auto-scaling, TLS/HTTPS, 30-day log retention |
| Edge deployment | Suno Sutra device | INT8/FP16 quantization, TensorRT/ONNX Runtime, fully offline |

---

## 4. Data Strategy

### 4.1 Dataset inventory

| Dataset | Size | Use | Gap/Limitation |
|---|---|---|---|
| ARTPARK-IISc VAANI | ~31,255 hrs audio (2,043 hrs transcribed) | ASR acoustic training | Severe imbalance: Bagri 0.63 hrs, Hadoti 0.34 hrs, Mewati 0.35 hrs |
| BPCC (IndicTrans2) | ~230M bitext pairs | MT encoder pre-training | No dialect-specific parallel text — scheduled languages only |
| IndicTTS Database | 272+ hrs, 13 languages | TTS vocoder training | Lumps dialects into one "Rajasthani" category |
| Speech-rj-hi (Karya) | 426,873 clips (~2.81 GB) | ASR/TTS augmentation | Read speech only, no dialect partitioning |
| LDC-IL Rajasthani Corpus | 31,096 words / 5,332 sentences | Golden MT evaluation set (kept out of training) | Too small to train from, restricted access |

### 4.2 Normalization pipeline (non-negotiable, precedes all training)
- Unicode NFC normalization across all text
- Nukta standardization: decompose precomposed characters (e.g. क़ U+0958) to base + modifier (क + ़ U+093C)
- Preserve retroflex lateral flap ळ (U+0933)
- Encode phonological shifts (e.g. Hindi /s/ → Marwari /h/) as mapping rules, not free-text edits

### 4.3 Augmentation (to offset low-resource dialects)
- **MT:** iterative back-translation using monolingual scrapes (forums, digitized literature, BhashaDaan donations) → pseudo-parallel pairs
- **ASR:** SpecAugment, speed perturbation (0.9x/1.0x/1.1x), background noise injection
- **Ongoing:** continuous ingestion from Bhashini BhashaDaan crowdsourcing

---

## 5. Roadmap (8 Months)

| Month | Phase | Key Activities | Milestone / Exit Criteria |
|---|---|---|---|
| 1–2 | Data curation & normalization | Ingest VAANI, BPCC, LDC-IL, Speech-rj-hi; run NFC/Nukta/ळ normalization; build BPE tokenizers | Clean, unified corpus + tokenizer ready |
| 3–4 | Base foundational training | Train FastConformer (ASR) + IndicTrans2 (MT) jointly on Hindi + Marwari (highest-resource dialect) | Working Hindi/Marwari baseline with acceptable WER |
| 5–6 | Dialect adaptation & augmentation | Freeze base encoders; fine-tune attention heads per dialect (Bagri, Hadoti, Mewati) using SpecAugment, back-translation, experience replay | All 6 dialects have fine-tuned checkpoints |
| 7 | Rigorous evaluation | ASR: CER; MT: chrF++ vs. LDC-IL golden set + COMET; TTS: MOS from native speaker blind tests | Evaluation scorecard signed off by linguists |
| 8 | Edge optimization & deployment | Prune/quantize (INT8) for Suno Sutra; deploy cloud APIs under Bhashini NHLT; enforce DPDP compliance (TLS, 30-day log retention) | Live cloud endpoint + offline edge build shipped |

---

## 6. Evaluation Framework

| System | Primary metric(s) | Why |
|---|---|---|
| ASR | Character Error Rate (CER) over WER | No standardized spelling for these dialects — CER reflects true phonetic accuracy |
| MT | chrF++ and COMET (not BLEU) | BLEU is unreliable for morphologically rich, low-resource languages |
| TTS | Mean Opinion Score (MOS) + Phonetically Balanced (PB) intelligibility tests | Naturalness requires native-speaker judgment |

Evaluation gate: MT scored against the LDC-IL corpus only (kept strictly out of training to avoid leakage).

---

## 7. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Cascading errors (ASR mistake propagates through MT/TTS) | High — destroys fidelity | Strong normalization + cross-lingual pre-training; monitor per-stage confidence |
| Intrasentential code-switching (Hindi↔dialect mid-sentence) | High — hallucinations at boundaries | Joint training on mixed-code data; targeted fine-tuning on code-switched samples |
| Shortage of expert linguists for deep dialects (Mewati, Bagri) | Medium — QA bottleneck | Prioritize linguist recruitment in Month 1; stagger dialect QA schedule |
| Extreme dialectal data imbalance | High — some dialects near-untrainable from scratch | Heavy reliance on cross-lingual transfer + augmentation (Section 4.3) |
| Edge hardware constraints (thermal/battery/memory on Suno Sutra) | Medium | INT8/FP16 quantization, TensorRT/ONNX compilation, ongoing compression |
| Orthographic inconsistency in crowdsourced transcripts | Medium — fragments tokenizer | Mandatory normalization pipeline before any training run |

---

## 8. Compliance & Security

- All processing localized within India per DPDP Act 2023 and Bhashini privacy specs
- No unauthorized secondary use of user-submitted data
- TLS/HTTPS encryption in transit
- Maximum 30-day log retention
- VAANI dataset usage requires contact-info sharing per its CC-BY-4.0 terms — track this obligation separately

---

## 9. Team & Collaboration Model

- **AI/ML engineers:** model architecture, training, quantization
- **Linguists (per dialect):** transcription QA, evaluation sign-off, phonological mapping validation
- **Data engineers:** normalization pipeline, corpus fusion, BhashaDaan ingestion
- Gate: no dialect fine-tune ships without linguist evaluation sign-off (Section 6)

---

## 10. Post-Roadmap / Future Direction

- Migrate from cascaded to end-to-end S2ST (e.g. SeamlessM4T-style) once parallel speech-to-speech data matures via BhashaDaan
- Layer in LLM + RAG grounded in Rajasthani cultural/governmental context for e-governance, healthcare, and education use cases beyond translation