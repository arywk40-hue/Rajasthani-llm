"""
MT Benchmark — real IndicTrans2 inference over the curated dialect pairs.

Loads IndicTrans2 via `src.mt.model.IndicTrans2MT`, translates each source sentence, and
scores the generated output against the reference with chrF++.

    PYTHONPATH=. python experiments/mt/benchmark.py \
        --output_csv results/mt/benchmark_results.csv

There is deliberately no placeholder path. If the model does not load, the script exits
non-zero without writing a CSV — a fabricated row is worse than no row.

An earlier version of this file set `hypotheses = src_texts`, scoring the untranslated
Devanagari source against the reference, and reported BLEU and COMET as `chrF * 0.8` and
`chrF * 0.95`. No model was loaded. Those derived columns are gone rather than renamed:
BLEU and COMET are not computed here, so they are not reported.

READ BEFORE CITING ANY NUMBER THIS PRODUCES
-------------------------------------------
The chrF++ values are now real — a real model's real output, scored by a real metric. That
does not make them a meaningful benchmark:

  * One sentence per dialect. A chrF++ mean over n=1 is an anecdote, not an evaluation.
  * All six dialects map to `hin_Deva` (src/mt/model.py:36-41), so the model cannot tell
    them apart. Differences between per-dialect rows reflect the sentences chosen, not
    dialect-specific model behaviour.

Both caveats are stamped into the CSV (`samples`, `lang_code`) and the report so a row
cannot be lifted out of context. Fixing them needs a genuine held-out eval set and a
fine-tuned dialect checkpoint — neither exists yet. See README "Known Limitations".
"""

import csv
import sys
import argparse
from pathlib import Path

from loguru import logger

from src.evaluation.metrics import Evaluator

DIALECTS = ["marwari", "mewari", "dhundhari", "hadoti", "mewati", "bagri"]
TARGET_LANGS = ["hindi", "english"]

# Marker that IndicTrans2MT.translate() returns when no HF model is loaded and the
# skeleton fallback is active (src/mt/model.py:291-292). Scoring these strings would
# produce a near-zero chrF++ that looks like a measurement of a bad model rather than
# the absence of one.
SKELETON_MARKER = "[SKELETON]"

# (source, hindi_reference, english_reference).
# These six sentences are hand-written for this benchmark, NOT drawn from data/linguistic/.
# They are the only per-dialect evaluation sentences that exist; data/linguistic/ has 12
# curated entries covering Marwari, Mewari, Dhundhari and Bagri only, with nothing for
# Hadoti or Mewati. Nothing here is a held-out split — the model has no dialect training
# data, so there is no train set for these to be held out from.
TEST_PAIRS = {
    "marwari": [("अठै सब चोखो है", "यहाँ सब ठीक है", "Everything is fine here")],
    "mewari": [("अणी तरफ आओ", "इस तरफ आओ", "Come this way")],
    "dhundhari": [("छोरा कठै जा रयो छै", "लड़का कहाँ जा रहा है", "Where is the boy going")],
    "hadoti": [("काय हाल छै", "क्या हाल है", "How are you")],
    "mewati": [("कहाँ जा रह्यो है", "कहाँ जा रहे हो", "Where are you going")],
    "bagri": [("किन्नै जावैगा", "किस तरफ जाओगे", "Which way will you go")],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MT benchmark across Rajasthani dialects")
    parser.add_argument("--output_csv", type=str, default="results/mt/benchmark_results.csv")
    parser.add_argument("--output_report", type=str, default="results/mt/benchmark_report.md")
    parser.add_argument("--model", type=str, default=None, help="Key from MODEL_REGISTRY or an HF id")
    args = parser.parse_args()

    from src.mt.model import IndicTrans2MT, _get_flores_code

    try:
        mt = IndicTrans2MT(model_name=args.model) if args.model else IndicTrans2MT()
    except Exception as e:
        logger.error(f"Could not construct IndicTrans2MT: {e}. No CSV written.")
        return 1

    evaluator = Evaluator()
    results = []

    for dialect in DIALECTS:
        samples = TEST_PAIRS.get(dialect, [])
        if not samples:
            logger.warning(f"No test sentences for {dialect} — omitted rather than zero-filled.")
            continue

        src_texts = [s[0] for s in samples]

        for tgt_idx, tgt_lang in enumerate(TARGET_LANGS, start=1):
            ref_texts = [s[tgt_idx] for s in samples]

            try:
                hypotheses = mt.translate(src_texts, src_lang=dialect, tgt_lang=tgt_lang)
            except Exception as e:
                logger.error(f"Translation failed for {dialect}->{tgt_lang}: {e}")
                return 1

            # The skeleton fallback returns "[SKELETON] Translation of: ..." rather than
            # raising, so an unloaded model would otherwise be scored and written out as
            # though it were measured. Refuse instead.
            if any(SKELETON_MARKER in h for h in hypotheses):
                logger.error(
                    "IndicTrans2 did not load — translate() returned skeleton placeholders. "
                    "This is not a measurement, so no CSV was written. Check network access "
                    "and that `transformers` can reach ai4bharat/indictrans2-*."
                )
                return 1

            empty = sum(1 for h in hypotheses if not h.strip())
            if empty:
                logger.warning(
                    f"{dialect}->{tgt_lang}: {empty}/{len(hypotheses)} translations came back "
                    "empty. These score chrF++ 0.0 and drag the mean down — check decoding "
                    "before reporting."
                )

            report = evaluator.evaluate_mt(hypotheses, ref_texts)

            results.append({
                "model": mt.model_id,
                "dialect": dialect,
                "target_lang": tgt_lang,
                "lang_code": _get_flores_code(dialect),
                "samples": len(samples),
                "empty_translations": empty,
                "chrf": round(report.mean_chrf, 4),
                "measured": "true",
            })
            logger.info(
                f"{dialect} -> {tgt_lang}: chrF++={results[-1]['chrf']} "
                f"over {len(samples)} sample(s)"
            )

    if not results:
        logger.error("No dialect produced a translation. No CSV written.")
        return 1

    fieldnames = [
        "model", "dialect", "target_lang", "lang_code",
        "samples", "empty_translations", "chrf", "measured",
    ]

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("# MT Benchmark — measured chrF++\n\n")
        f.write(
            f"Model: `{results[0]['model']}`. Hypotheses are generated by IndicTrans2 and "
            "scored against the references with chrF++.\n\n"
            "**These numbers are real but not meaningful as a dialect benchmark.** Each row "
            "covers a single sentence, and every dialect resolves to the same model language "
            "code (`hin_Deva`), so the model cannot distinguish them. Differences between "
            "rows reflect the chosen sentences, not dialect-specific behaviour. BLEU and "
            "COMET are not computed and so are not reported.\n\n"
        )
        f.write("| Model | Dialect | Direction | Lang code | n | chrF++ |\n")
        f.write("| :--- | :--- | :--- | :--- | ---: | ---: |\n")
        for r in results:
            f.write(
                f"| {r['model']} | {r['dialect'].capitalize()} | {r['dialect']} → "
                f"{r['target_lang']} | {r['lang_code']} | {r['samples']} | {r['chrf']} |\n"
            )

    covered = {r["dialect"] for r in results}
    if missing := [d for d in DIALECTS if d not in covered]:
        logger.warning(f"No measurement for: {missing}. Report coverage with any figure.")

    logger.success(f"Wrote {len(results)} measured rows to {out_csv} and {out_report}")
    logger.warning(
        "n=1 per row and all dialects share hin_Deva — cite these as a smoke test of the "
        "translation path, not as dialect MT quality."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
