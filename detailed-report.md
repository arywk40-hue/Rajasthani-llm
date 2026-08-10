# Detailed Report: Rajasthani Dialect AI

This document provides a comprehensive overview of the Rajasthani Dialect AI project, as required for the Bhashini Hackathon submission.

## 1. Problem Statement

The goal of this project is to develop robust AI language technologies (ASR, MT, and TTS) for six primary Rajasthani dialects: Marwari, Mewari, Dhundhari, Hadoti, Mewati, and Bagri. These dialects historically lack open-source models, and our aim is to bridge this gap by fine-tuning existing Indic foundation models on carefully filtered dialect datasets.

## 2. Architecture and Approach

We adopted a 3-track strategy to build a production-grade system optimized for edge deployment.

**Implementation status: all three tracks are architecture only. No model has been trained
and `models/checkpoints/` is empty.** The wrappers, trainers and samplers described below
are real, working code awaiting data.

*   **ASR (Automatic Speech Recognition):** We use `vasista22/whisper-hindi-large-v2` (IndicWhisper) as the base model. To adapt it to dialects without catastrophic forgetting, we freeze the encoder and fine-tune only the decoder. *Wrapper and `Seq2SeqTrainer` integration implemented; never trained (no audio).*
*   **MT (Machine Translation):** We use `AI4Bharat/IndicTrans2` (1B and 200M variants) leveraging model souping (Wortsman et al.) and experience replay. This mixes 15-20% of general domain data (e.g., BPCC) during training to retain baseline Hindi/English translation quality while learning dialectal nuances. *Souping, experience replay and back-translation implemented; no dialect checkpoint exists, and BPCC has not been fetched.*
*   **TTS (Text-to-Speech):** Due to the lack of open weights for `ai4bharat/indic-tts-rajasthani-fastpitch`, the TTS subsystem routes synthesis requests to the Bhashini TTS API. *This path requires a valid API key and a confirmed service ID. Five Rajasthani dialects have been added to the Bhashini platform, but dialect enablement can reach ASR before TTS, so the TTS service ID must be verified against the platform's model catalogue. The local fallback `src/tts/hifigan.py` is a 5-layer placeholder, not a real HiFi-GAN vocoder.*

## 3. Data Strategy

The intended sources are below. **Status: the fetch code exists but has not been run to
completion — `data/raw/vaani/` is empty and no audio file is present in this repository.**

*   **VAANI (ARTPARK-IISc):** Filter the ~31,000-hour corpus (of which ~2,043 hours are
    transcribed) to Rajasthan districts, mapping district metadata onto the 6 target
    dialects. *Not yet fetched.*
*   **Karya (`severo/speech-rj-hi`):** 426,873 read-speech recordings from 98 participants
    in Soda, Rajasthan. *Currently cached as 10,000 text-only rows with no `audio_path`
    field; all rows carry the single label `rajasthani`, so no per-dialect split exists.
    The text is standard Hindi register — it is Rajasthani-accented Hindi speech, which
    suits ASR acoustics but is not a source of dialect text.*

We employ rigorous pre-processing, including the `DevanagariNormalizer`, to preserve
critical phonetic characters like Nukta and ळ, which are prevalent in Rajasthani scripts.
This component is implemented and unit-tested.

### Known constraint: dialect codes

IndicTrans2 covers only the 22 scheduled languages of India; Rajasthani and its varieties
are not among them. All six dialects therefore map to `hin_Deva` in `src/mt/model.py`,
meaning the base model cannot distinguish them from Hindi or from each other. Per-dialect
MT figures are not meaningful until a fine-tuned checkpoint exists.

## 4. Evaluation and Metrics

Our evaluation pipeline (`src/evaluation/metrics.py`) implements every metric in pure
Python with no external metric dependency, so it runs deterministically on any machine:

*   **ASR:** Word Error Rate (WER) and Character Error Rate (CER) via a Wagner-Fischer
    edit-distance implementation. CER is the primary metric, because these dialects have
    no standardised orthography. Not `jiwer` — there is no such dependency.
*   **MT:** chrF++ (character n-grams to order 6 plus word n-grams to order 2, β=2) via a
    local implementation. Not `sacrebleu`. **BLEU is not implemented**; it was
    deliberately avoided for morphologically rich low-resource dialects. COMET is an
    optional wrapper requiring `unbabel-comet` and is not installed by default.

**No metric has yet been run against real model output.** The scripts under
`experiments/` emit placeholder values — every artifact they write is named
`*.PLACEHOLDER.*`. See the Benchmark Status section of `README.md`.

## 5. Edge Deployment Readiness

The pipeline includes an INT8/FP16 quantization flow (`src/edge/quantizer.py`) using ONNX Runtime, designed to compress the IndicTrans2 200M model and Whisper small models to fit within a strict 900MB RAM budget for mobile and edge devices.