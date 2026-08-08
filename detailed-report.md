# Detailed Report: Rajasthani Dialect AI

This document provides a comprehensive overview of the Rajasthani Dialect AI project, as required for the Bhashini Hackathon submission.

## 1. Problem Statement

The goal of this project is to develop robust AI language technologies (ASR, MT, and TTS) for six primary Rajasthani dialects: Marwari, Mewari, Dhundhari, Hadoti, Mewati, and Bagri. These dialects historically lack open-source models, and our aim is to bridge this gap by fine-tuning existing Indic foundation models on carefully filtered dialect datasets.

## 2. Architecture and Approach

We adopted a 3-track strategy to build a production-grade system optimized for edge deployment:

*   **ASR (Automatic Speech Recognition):** We use `vasista22/whisper-hindi-large-v2` (IndicWhisper) as the base model. To adapt it to dialects without catastrophic forgetting, we freeze the encoder and fine-tune only the decoder.
*   **MT (Machine Translation):** We use `AI4Bharat/IndicTrans2` (1B and 200M variants) leveraging model souping (Wortsman et al.) and experience replay. This mixes 15-20% of general domain data (e.g., BPCC) during training to retain baseline Hindi/English translation quality while learning dialectal nuances.
*   **TTS (Text-to-Speech):** Due to the lack of open-weights for `ai4bharat/indic-tts-rajasthani-fastpitch`, we have architected the TTS subsystem to securely route synthesis requests to the Bhashini TTS API endpoint, providing a seamless fallback for missing local models.

## 3. Data Strategy

We source high-quality datasets directly from HuggingFace, drastically reducing the need for local storage:

*   **VAANI (ARTPARK-IISc):** We filter the 31,000-hour corpus strictly to the 29 districts of Rajasthan, dynamically mapping the district metadata to our 6 target dialects.
*   **Karya (speech-rj-hi):** We stream over 426,000 audio recordings of read speech for robust ASR benchmarking and TTS fine-tuning.

We employ rigorous pre-processing, including the `DevanagariNormalizer`, to preserve critical phonetic characters like Nukta and ळ, which are prevalent in Rajasthani scripts.

## 4. Evaluation and Metrics

Our evaluation pipeline (`src/evaluation/metrics.py`) supports:
*   **ASR:** Word Error Rate (WER) and Character Error Rate (CER) via the `jiwer` library.
*   **MT:** chrF++ and BLEU scores via `sacrebleu`, tracking both semantic preservation and exact matching.

## 5. Edge Deployment Readiness

The pipeline includes an INT8/FP16 quantization flow (`src/edge/quantizer.py`) using ONNX Runtime, designed to compress the IndicTrans2 200M model and Whisper small models to fit within a strict 900MB RAM budget for mobile and edge devices.