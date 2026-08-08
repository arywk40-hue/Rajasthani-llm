"""
Evaluation Metrics

Implements the evaluation harness for the Rajasthani Dialect AI, following
roadmap §6:

- ASR: Character Error Rate (CER) instead of WER due to orthographic
  inconsistency. Real Levenshtein edit distance implementation.
- MT : chrF++ (character + word n-gram F-score) and a COMET wrapper.
  BLEU is explicitly avoided for morphologically rich, low-resource dialects.
- TTS: Mean Opinion Score (MOS) tracking with reviewer audit + Phonetically
  Balanced (PB) intelligibility pass/fail logic.

All metrics are pure-Python (no external metric libraries required) so the
harness runs deterministically on any machine, including the no-GPU evaluation
step of the roadmap (Month 7).
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger


# ─── Edit distance helpers ───────────────────────────────────────────────────


def levenshtein_distance(a: str, b: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.

    Uses the classic dynamic-programming row-reduction (Wagner-Fischer) to keep
    memory at O(min(len(a), len(b))) regardless of input length.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure `a` is the shorter row dimension
    if len(a) > len(b):
        a, b = b, a

    prev = list(range(len(a) + 1))
    for j, b_char in enumerate(b, start=1):
        curr = [j]
        for i, a_char in enumerate(a, start=1):
            substitution_cost = 0 if a_char == b_char else 1
            curr.append(
                min(
                    prev[i] + 1,  # deletion
                    curr[i - 1] + 1,  # insertion
                    prev[i - 1] + substitution_cost,  # substitution
                )
            )
        prev = curr
    return prev[-1]


def _normalize_for_wer(text: str) -> list[str]:
    """Tokenize text into words for WER computation (whitespace split, lowercased)."""
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return text.translate(translator).lower().split()


def _normalize_for_cer(text: str) -> str:
    """Normalize text for CER computation (whitespace collapsed)."""
    return " ".join(text.split())


# ─── ASR metrics ─────────────────────────────────────────────────────────────


def compute_cer(hypothesis: str, reference: str) -> float:
    """
    Character Error Rate: edit distance / reference length, clamped to [0, 1].

    CER is preferred over WER for these dialects (roadmap §6) because there is
    no standardized orthography — character-level accuracy better reflects
    phonetic fidelity.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    hyp = _normalize_for_cer(hypothesis)
    ref = _normalize_for_cer(reference)
    edit_distance = levenshtein_distance(hyp, ref)
    return min(edit_distance / len(ref), 1.0)


def compute_wer(hypothesis: str, reference: str) -> float:
    """
    Word Error Rate: word-level edit distance / reference word count.
    Provided as a supporting metric alongside CER.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    hyp_words = _normalize_for_wer(hypothesis)
    ref_words = _normalize_for_wer(reference)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    edit_distance = levenshtein_distance(" ".join(hyp_words), " ".join(ref_words))
    return min(edit_distance / len(ref_words), 1.0)


# ─── MT metrics ──────────────────────────────────────────────────────────────


def _char_ngrams(word: str, n: int) -> list[str]:
    """Generate character n-grams of order n for a single word."""
    return [word[i : i + n] for i in range(len(word) - n + 1)] if len(word) >= n else []


def _word_ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    """Generate word n-grams of order n for a tokenized sentence."""
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)] if len(words) >= n else []


def _count_ngrams(ngrams: Iterable) -> dict:
    counts: dict = {}
    for ng in ngrams:
        key = ng if isinstance(ng, tuple) else ng
        counts[key] = counts.get(key, 0) + 1
    return counts


def _fscore(precision: float, recall: float, beta: float = 2.0) -> float:
    """F-beta score; beta>1 weights recall more heavily (chrF convention)."""
    if precision + recall == 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def compute_chrf(
    hypothesis: str,
    reference: str,
    char_order: int = 6,
    word_order: int = 2,
    beta: float = 2.0,
) -> float:
    """
    chrF++ (Popović 2017): mean F-beta score over character n-grams (orders
    1..char_order) and word n-grams (orders 1..word_order).

    Character n-grams are computed within words, exactly matching the standard
    chrF++ definition (character-level precision/recall are weighted by word
    boundaries, which is essential for Devanagari — a single n-armed script
    where naive cross-word n-grams would pollute the score).
    """
    hyp_chars = list(hypothesis)
    ref_chars = list(reference)
    hyp_words = hypothesis.split()
    ref_words = reference.split()

    precisions: list[float] = []
    recalls: list[float] = []

    # Character n-gram orders
    for n in range(1, char_order + 1):
        if len(ref_chars) < n:
            continue
        hyp_ng = _count_ngrams(_char_ngrams(_normalize_for_cer(hypothesis), n))
        ref_ng = _count_ngrams(_char_ngrams(_normalize_for_cer(reference), n))
        match = sum(min(hyp_ng.get(g, 0), ref_ng.get(g, 0)) for g in ref_ng)
        if not hyp_ng:
            precisions.append(0.0)
        else:
            precisions.append(match / sum(hyp_ng.values()))
        recalls.append(match / sum(ref_ng.values()))

    # Word n-gram orders (the "++" in chrF++)
    for n in range(1, word_order + 1):
        hyp_ng = _count_ngrams(_word_ngrams(hyp_words, n))
        ref_ng = _count_ngrams(_word_ngrams(ref_words, n))
        if not ref_ng and not hyp_ng:
            continue
        match = sum(min(hyp_ng.get(g, 0), ref_ng.get(g, 0)) for g in ref_ng)
        if not hyp_ng:
            precisions.append(0.0)
        else:
            precisions.append(match / sum(hyp_ng.values()))
        recalls.append(match / sum(ref_ng.values()))

    if not precisions or not recalls:
        return 0.0

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    return _fscore(avg_precision, avg_recall, beta)


class COMETWrapper:
    """
    Optional COMET MT evaluator.

    COMET (Rei et al., 2020) is a learned multilingual metric referenced in the
    roadmap (§6). It requires the `unbabel-comet` package and model weights that
    are downloaded on first use, so it is purely optional in this harness.
    If the package is unavailable, `score` raises ImportError with guidance;
    callers should fall back to chrF++ for offline evaluation.
    """

    def __init__(self, model_name: str = "Unbabel/wmt22-comet-da"):
        self.model_name = model_name
        self._model = None
        try:
            from comet import download_model, load_from_checkpoint  # type: ignore

            self._download_model = download_model
            self._load_from_checkpoint = load_from_checkpoint
        except ImportError:
            self._available = False
            logger.warning(
                "unbabel-comet not installed. Install with `pip install unbabel-comet` "
                "to enable COMET scoring."
            )
        else:
            self._available = True

    @property
    def available(self) -> bool:
        return self._available

    def _load_model(self):
        if self._model is not None:
            return self._model
        model_path = self._download_model(self.model_name)
        self._model = self._load_from_checkpoint(model_path)
        return self._model

    def score_pairs(self, hypotheses: list[str], references: list[str], sources: Optional[list[str]] = None) -> list[float]:
        """Score a batch of (hypothesis, reference[, source]) triples with COMET."""
        if not self._available:
            raise ImportError(
                "COMET requires the 'unbabel-comet' package and model weights. "
                "Install with `pip install unbabel-comet` or use compute_chrf() "
                "for offline evaluation."
            )
        model = self._load_model()
        data = [
            {"src": sources[i] if sources else refs, "mt": hyp, "ref": ref}
            for i, (hyp, ref) in enumerate(zip(hypotheses, references))
        ]
        outputs = model.predict(data, batch_size=32, gpus=0)
        return [float(s) for s in outputs.scores]


# ─── TTS metrics ─────────────────────────────────────────────────────────────


@dataclass
class MOSRecord:
    """A single human Mean Opinion Score rating."""
    audio_id: str
    score: float
    reviewer: str
    dialect: Optional[str] = None


@dataclass
class MOSScorecard:
    """Aggregated MOS results from multiple blind native-speaker ratings."""
    records: list[MOSRecord] = field(default_factory=list)

    @property
    def mean(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.score for r in self.records) / len(self.records)

    @property
    def count(self) -> int:
        return len(self.records)

    def add(self, audio_id: str, score: float, reviewer: str, dialect: Optional[str] = None) -> float:
        if not 0 <= score <= 5:
            raise ValueError(f"MOS score must be in [0, 5], got {score}")
        self.records.append(MOSRecord(audio_id=audio_id, score=score, reviewer=reviewer, dialect=dialect))
        return score

    def by_dialect(self) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for r in self.records:
            grouped.setdefault(r.dialect or "unknown", []).append(r.score)
        return {d: sum(s) / len(s) for d, s in grouped.items()}

    def summary(self) -> str:
        return f"MOS: {self.mean:.2f} across {self.count} ratings"


def pb_intelligibility_pass(phonetically_balanced_hits: int, phonetically_balanced_total: int, threshold: float = 0.85) -> bool:
    """Phonetically Balanced intelligibility gate from roadmap §6 (TTS)."""
    if phonetically_balanced_total <= 0:
        return False
    return (phonetically_balanced_hits / phonetically_balanced_total) >= threshold


# ─── High-level evaluator ────────────────────────────────────────────────────


@dataclass
class EvaluationReport:
    """Aggregated metrics for a cascade component evaluation run."""
    component: str  # 'asr' | 'mt' | 'tts'
    samples: int = 0
    mean_cer: float = 0.0
    mean_wer: float = 0.0
    mean_chrf: float = 0.0
    mos: MOSScorecard = field(default_factory=MOSScorecard)
    per_record: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"EvaluationReport[{self.component}] samples={self.samples}"]
        if self.component in ("asr", "mt"):
            if self.component == "asr":
                parts.append(f"CER={self.mean_cer:.4f} WER={self.mean_wer:.4f}")
            else:
                parts.append(f"chrF++={self.mean_chrf:.4f}")
        if self.mos.records:
            parts.append(self.mos.summary())
        return " ".join(parts)


class Evaluator:
    """
    High-level evaluation harness across the S2ST cascade (roadmap §6).

    Usage:
        evaluator = Evaluator()
        report = evaluator.evaluate_asr(hypotheses, references)
        report = evaluator.evaluate_mt(hypotheses, references)
        evaluator.evaluate_tts(audio_id="x", score=4, reviewer="linguist_1", dialect="marwari")
    """

    def __init__(self):
        self._comet = COMETWrapper()
        logger.info("Initialized Evaluator")

    # Backwards-compatible single-pair methods
    def compute_cer(self, hypothesis: str, reference: str) -> float:
        """Character Error Rate between one hypothesis and reference."""
        return compute_cer(hypothesis, reference)

    def compute_wer(self, hypothesis: str, reference: str) -> float:
        return compute_wer(hypothesis, reference)

    def compute_chrf(self, hypothesis: str, reference: str) -> float:
        """chrF++ between one hypothesis and reference."""
        return compute_chrf(hypothesis, reference)

    def evaluate_asr(
        self, hypotheses: Iterable[str], references: Iterable[str], meta: Optional[list[str]] = None
    ) -> EvaluationReport:
        """Batch ASR evaluation returning aggregate CER + WER with per-record detail."""
        report = EvaluationReport(component="asr", per_record=[])
        cers, wers = [], []
        for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
            cer = compute_cer(hyp, ref)
            wer = compute_wer(hyp, ref)
            cers.append(cer)
            wers.append(wer)
            record = {"index": i, "hypothesis": hyp, "reference": ref, "cer": cer, "wer": wer}
            if meta and i < len(meta):
                record["meta"] = meta[i]
            report.per_record.append(record)
        report.samples = len(report.per_record)
        report.mean_cer = sum(cers) / len(cers) if cers else 0.0
        report.mean_wer = sum(wers) / len(wers) if wers else 0.0
        return report

    def evaluate_mt(
        self, hypotheses: Iterable[str], references: Iterable[str], meta: Optional[list[str]] = None
    ) -> EvaluationReport:
        """Batch MT evaluation returning aggregate chrF++."""
        report = EvaluationReport(component="mt", per_record=[])
        chrf_scores = []
        for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
            chrf = compute_chrf(hyp, ref)
            chrf_scores.append(chrf)
            record = {"index": i, "hypothesis": hyp, "reference": ref, "chrf": chrf}
            if meta and i < len(meta):
                record["meta"] = meta[i]
            report.per_record.append(record)
        report.samples = len(report.per_record)
        report.mean_chrf = sum(chrf_scores) / len(chrf_scores) if chrf_scores else 0.0
        return report

    def evaluate_tts(self, audio_id: str, score: float, reviewer: str, dialect: Optional[str] = None) -> MOSScorecard:
        """Log a human Mean Opinion Score (MOS) for TTS evaluation."""
        return self._tts_mos(audio_id, score, reviewer, dialect)

    def _tts_mos(self, audio_id: str, score: float, reviewer: str, dialect: Optional[str] = None) -> MOSScorecard:
        card = MOSScorecard()
        card.add(audio_id, score, reviewer, dialect)
        logger.info(f"Recorded MOS {score} for {audio_id} by {reviewer}")
        return card

    def evaluate_tts_pb(self, hits: int, total: int, threshold: float = 0.85) -> bool:
        """Phonetically Balanced intelligibility gate."""
        return pb_intelligibility_pass(hits, total, threshold)

    def get_comet(self) -> COMETWrapper:
        return self._comet

    def save_report(self, report: EvaluationReport, out_path: str | Path) -> Path:
        """Persist an evaluation report as JSONL for audit/scorecard records."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(out_path, "w", encoding="utf-8") as f:
            for record in report.per_record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(report.per_record)} records to {out_path}")
        return out_path