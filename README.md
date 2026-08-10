# Rajasthani Language AI — Multilingual Speech & Translation Platform

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](pyproject.toml)

An end-to-end, evidence-driven AI framework for low-resource **Rajasthani dialects**: Marwari, Mewari, Dhundhari, Hadoti, Mewati, and Bagri. Provides Automatic Speech Recognition (ASR), Machine Translation (MT), and Text-to-Speech (TTS) capabilities.

---

## Current Status

**No model has been trained on dialect data, and no benchmark number here reflects
dialect-specific model quality.** `models/checkpoints/` is empty.

Of the four scripts under `experiments/`, two now run real inference and two still emit
placeholders:

- **ASR** and **MT** load real pretrained models, generate output, and score it. Each exits
  non-zero without writing a file rather than falling back to a placeholder. ASR needs audio
  that has not been fetched yet, so it currently produces nothing; MT runs, but on one
  sentence per dialect.
- **TTS** and **end-to-end** measure nothing. MOS is a human rating and no listening study
  has been run; the end-to-end script writes `NOT_RUN` without invoking the pipeline.

The evaluation *metric implementations* in `src/evaluation/metrics.py` are real and verified.
See [Benchmark Status](#benchmark-status) for what each script actually computes.

Read this section before citing anything below it.

---

## Problem Statement Traceability Matrix

Status values: **Implemented** (runs, produces real output) · **Code only** (implementation
exists, never executed against real data) · **Partial** · **Not measured** · **Docs only**

| Problem Requirement | Implementation | Evidence Artifact / Location | Status |
| :--- | :--- | :--- | :--- |
| **Benchmark MT Models** | Multi-dialect MT benchmarking suite | [`experiments/mt/benchmark.py`](experiments/mt/benchmark.py) | **Implemented (harness) / Smoke test only (result)** — runs real IndicTrans2 inference and scores generated output with chrF++. One sentence per dialect and a shared `hin_Deva` code mean the figures test the path, not dialect quality |
| **Collect Speech Data** | VAANI & Karya data fetching & validation pipeline | [`scripts/fetch_data.py`](scripts/fetch_data.py), [`scripts/data/validate_audio.py`](scripts/data/validate_audio.py) | **Code only** — `data/raw/vaani/` is empty; the Karya JSONL is text-only with no `audio_path` field, so zero audio is available for training |
| **Collect Text Data** | Text extraction, cleaning, and normalization pipeline | [`scripts/data/normalize_text.py`](scripts/data/normalize_text.py), [`scripts/data/validate_text.py`](scripts/data/validate_text.py) | **Partial** — 10,000 lines cached, all labelled `dialect: "rajasthani"`, in standard Hindi register (see [Data Inventory](#data-inventory-actual)) |
| **Six Dialects Coverage** | Inventory & metadata for Marwari, Mewari, Dhundhari, Hadoti, Mewati, Bagri | [`docs/datasets/dataset_inventory.md`](docs/datasets/dataset_inventory.md) | **Docs only** — no per-dialect data, checkpoints, or evaluation splits exist; all six dialects resolve to the same model language code (see [Known Limitations](#known-limitations)) |
| **Build ASR System** | Whisper / IndicWhisper fine-tuning architecture | [`src/asr/model.py`](src/asr/model.py), [`src/asr/trainer.py`](src/asr/trainer.py) | **Code only** — real `WhisperForConditionalGeneration` wrapper with encoder freezing and checkpoint management; never trained (no audio data) |
| **Build TTS System** | FastPitch + Bhashini API fallback speech synthesis | [`src/tts/fastpitch.py`](src/tts/fastpitch.py) | **Partial** — Bhashini cloud path requires an API key; `src/tts/hifigan.py` is a 5-layer placeholder, not HiFi-GAN; no local model |
| **Build MT System** | IndicTrans2 translation with model souping | [`src/mt/model.py`](src/mt/model.py), [`src/mt/trainer.py`](src/mt/trainer.py) | **Code only** — wrapper, experience replay, model souping and back-translation are real implementations; no dialect checkpoint exists |
| **Preserve Nuances** | Dedicated datasets for idioms, cultural expressions, and code-switching | [`data/linguistic/`](data/linguistic/) | **Partial** — 12 curated entries total (3 idioms, 3 cultural expressions, 4 dialect terms, 2 code-switching). The Devanagari normalizer is production-grade |
| **WER Evaluation** | Character & Word Error Rate evaluation harness | [`src/evaluation/metrics.py`](src/evaluation/metrics.py), [`experiments/asr/benchmark.py`](experiments/asr/benchmark.py) | **Implemented (metric and harness) / Not measured (no audio)** — Wagner-Fischer edit distance is correct and unit-tested, and the benchmark now runs real IndicWhisper inference. It writes no CSV until audio is fetched, so no results file exists |
| **MOS Evaluation** | Mean Opinion Score & Phonetically Balanced (PB) intelligibility gate | [`src/evaluation/metrics.py`](src/evaluation/metrics.py), [`results/tts_results.PLACEHOLDER.csv`](results/tts_results.PLACEHOLDER.csv) | **Not measured** — MOS requires human raters; no listening study has been run. The CSV values are hardcoded constants |
| **Use-Case Coverage** | Agriculture, Healthcare, Government, Education scenarios | [`experiments/end_to_end/benchmark.py`](experiments/end_to_end/benchmark.py), [`results/end_to_end_results.PLACEHOLDER.csv`](results/end_to_end_results.PLACEHOLDER.csv) | **Not measured** — the script writes `NOT_RUN` for every stage without invoking the pipeline; no domain test sets exist |
| **Privacy & Security** | Data governance, DPDP compliance, & encryption docs | [`docs/privacy.md`](docs/privacy.md), [`docs/security.md`](docs/security.md), [`docs/data-governance.md`](docs/data-governance.md) | **Partial** — DPDP middleware sets audit headers; API-key check is trivially bypassable and there is no rate limiting |

---

## System Architecture

```text
                    ┌──────────────────────┐
                    │   User Speech Input   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │         ASR          │
                    │  Speech → Rajasthani │
                    │       Text           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Text Normalization   │
                    │ Script + Orthography │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │         MT           │
                    │ Rajasthani → Hindi   │
                    │ Rajasthani → English │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │         TTS          │
                    │ Text → Target Speech │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Audio Response    │
                    └──────────────────────┘
```

---

## Benchmark Status

Two of the four scripts under `experiments/` now run real inference; two still emit
placeholder output. This table records exactly what each one computes so the artifacts in
`results/` are not mistaken for measurements.

| Benchmark | Script | What it actually does | Valid? |
| :--- | :--- | :--- | :--- |
| **MT** | `experiments/mt/benchmark.py` | **Rewritten to run real inference.** Loads IndicTrans2, translates each source sentence and scores the output with chrF++. Refuses to write a CSV if the model falls back to skeleton mode. Numbers are real but rest on one sentence per dialect, and all six dialects share `hin_Deva` — a smoke test of the translation path, not dialect MT quality. | Yes, as a smoke test |
| **ASR** | `experiments/asr/benchmark.py` | **Rewritten to run real inference.** Loads IndicWhisper, transcribes the audio listed in the fetcher's manifest, and computes CER/WER per sample plus RTF from measured wall time over measured audio duration. Exits non-zero without writing a CSV when no audio is present — which is the current state. | Yes, once audio exists |
| **TTS** | `experiments/tts/benchmark.py` | Passes the constant `4.2` into `evaluate_tts()` and reads it back. `pb_pass` is hardcoded `hits=9, total=10`. MOS is by definition a human rating; no listening study has been conducted. | No |
| **End-to-end** | `experiments/end_to_end/benchmark.py` | Writes `NOT_RUN` for the ASR, MT, TTS and end-to-end status of four use-case rows, with latency and success rate left blank. The pipeline is never invoked. Earlier revisions wrote `PASS`, `850` and `95%` as literals here. | No |

### Reproducing the current MT figures

`experiments/mt/benchmark.py` now loads IndicTrans2 and translates before scoring, so the
chrF++ column is a genuine measurement of generated output. Two caveats travel with every
number it writes, and are stamped into the CSV (`samples`, `lang_code`) so a row cannot be
quoted without them:

- **n=1 per row.** Each dialect contributes a single sentence, so a per-dialect mean is an
  anecdote, not an evaluation.
- **All six dialects resolve to `hin_Deva`.** The model cannot tell them apart, so
  differences between rows reflect the sentences chosen, not dialect-specific behaviour.

```bash
PYTHONPATH=. python experiments/mt/benchmark.py
# Downloads the IndicTrans2 checkpoint on first run. Exits 1 without writing a CSV if the
# model cannot load, rather than scoring skeleton placeholders as if they were output.
```

Treat the result as a smoke test of the translation path. A real MT baseline needs a
held-out eval set and a fine-tuned dialect checkpoint; neither exists yet.

Earlier revisions of this script scored the untranslated source against the reference and
derived BLEU as `chrF × 0.8` and COMET as `chrF × 0.95`. Those derived columns were
removed rather than renamed — BLEU and COMET are not computed, so they are not reported.

### What a valid baseline requires

| Benchmark | Blocked on |
| :--- | :--- |
| ASR (zero-shot CER) | Real dialect audio. The harness is done — `experiments/asr/benchmark.py` runs IndicWhisper and computes CER/WER/RTF from the fetcher manifest. Run `scripts/fetch_data.py --with-audio` and it produces measurements. **Nearest to achievable.** |
| MT (chrF++) | Genuine dialect↔Hindi parallel pairs. The training file now holds 16 real pairs and nothing else — enough to exercise the trainer, far too few to fine-tune on |
| TTS (MOS) | A working local model, then a blind native-speaker listening study |
| End-to-end | Per-domain test sets and a wired ASR→MT→TTS path |

---

## Data Inventory (Actual)

| Source | Claimed | On disk |
| :--- | :--- | :--- |
| VAANI speech | Spontaneous speech, 165 districts | **Empty** — `data/raw/vaani/` contains no files |
| Karya speech | 426,000+ audio clips | 10,000 JSONL lines, **text only**; keys are `text`, `dialect`, `source`, `sample_rate` — there is no `audio_path`, so none of it is usable for ASR training |
| Dialect labels | Six distinct dialects | All 10,000 rows are labelled `dialect: "rajasthani"` — no per-dialect split exists |
| Text register | Rajasthani dialect text | Standard Hindi. Sample: `यह उसी की पुण्य-स्मृति है।` The upstream dataset is 98 speakers from Soda, Rajasthan reading stories aloud, i.e. Rajasthani-accented Hindi *speech* — useful for ASR acoustics, not a source of dialect text |
| AI4Bharat BPCC parallel | Rajasthani ↔ Hindi/English | Not fetched |
| MT training pairs | — | 16 rows in `data/processed/mt_mps_train.jsonl` — the 8 usable curated entries, each in both directions. Previously 516 rows, but 500 of those had `source_text` identical to `target_text`; the generator block that produced them has been removed from `scripts/train_local_mps.py` |
| Curated linguistic sets | Idioms, cultural expressions, code-switching | 12 entries total |
| Trained checkpoints | — | None; `models/checkpoints/` is 0 bytes |

---

## Known Limitations

### All six dialects share one model language code

IndicTrans2 covers exactly the 22 scheduled languages of India. Rajasthani is not among
them, and neither are any of its varieties. Consequently `src/mt/model.py` maps every
dialect onto Hindi:

```python
FLORES_LANG_CODES = {
    "marwari": "hin_Deva",   # Closest proxy — no dedicated FLORES code for dialects
    "mewari": "hin_Deva",
    "dhundhari": "hin_Deva",
    "hadoti": "hin_Deva",
    "mewati": "hin_Deva",
    "bagri": "hin_Deva",
}
```

**Implication:** the model cannot distinguish the six dialects from one another or from
Hindi. Any per-dialect MT score reported before fine-tuning reflects differences between
the evaluation sentences, not dialect-specific model behaviour. Reporting per-dialect
columns without this caveat is misleading.

This is a property of the available pretrained models, not a defect in this codebase. The
documented path forward is to fine-tune from the `hin_Deva` direction on genuine dialect
parallel data. Fed dialect text under `hin_Deva`, IndicTrans2 will not error — it degrades
silently, worst on dialect-specific vocabulary, postpositions, and verb morphology.

### Other constraints

- **No trained model exists.** Everything under `src/` is architecture awaiting data.
- **`src/tts/hifigan.py` is not HiFi-GAN** — a 5-layer `Conv1d` stack standing in for the real vocoder.
- **TTS depends on Bhashini cloud access.** Five Rajasthani dialects have been added to the Bhashini platform, but dialect enablement can land in ASR before TTS; confirm a callable TTS service ID before relying on this path.
- **API-key authentication is bypassable** and there is no rate limiting. Not deployment-ready.

---

## Quickstart & Installation

```bash
# 1. Clone repository
git clone https://github.com/arywk40-hue/Rajasthani-llm.git
cd Rajasthani-llm

# 2. Install dependencies
pip install -r requirements.txt

# 3. Validate dataset quality
python scripts/data/validate_text.py --input data/linguistic/idioms.json

# 4. Fetch audio (required before the ASR benchmark can produce anything)
python scripts/fetch_data.py --with-audio --max-vaani 200

# 5. Run the ASR benchmark — real inference; exits 1 with no CSV if step 4 was skipped
PYTHONPATH=. python experiments/asr/benchmark.py

# 6. Run the MT benchmark — real IndicTrans2 inference, downloads the checkpoint on first
#    run. Exits 1 without a CSV if the model cannot load. See the caveats above: n=1 per
#    dialect and a shared hin_Deva code make this a smoke test, not a quality measure.
PYTHONPATH=. python experiments/mt/benchmark.py

# 7. Placeholder benchmarks. These emit *.PLACEHOLDER.* artifacts and measure nothing;
#    they exist to be replaced, not to be cited. See Benchmark Status above.
PYTHONPATH=. python experiments/tts/benchmark.py
PYTHONPATH=. python experiments/end_to_end/benchmark.py
```
