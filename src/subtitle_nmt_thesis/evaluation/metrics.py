from __future__ import annotations

import json
from pathlib import Path

import sacrebleu
from comet import download_model, load_from_checkpoint
from subtitle_nmt_thesis.data.parser import parse_srt
from subtitle_nmt_thesis.evaluation.subtitle_metrics import SubtitleMetrics


try:
    _COMET_MODEL = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
except Exception:
    _COMET_MODEL = None


def sacrebleu_score(references: list[str], hypotheses: list[str]) -> dict:
    refs = [[r] for r in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    return {"bleu": bleu.score, "precisions": bleu.precisions}


def chrf_score(references: list[str], hypotheses: list[str]) -> dict:
    refs = [[r] for r in references]
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    return {"chrf": chrf.score}


def comet_score(references: list[str], hypotheses: list[str], src_texts: list[str]) -> dict:
    if _COMET_MODEL is None:
        return {"comet": None, "skipped": True}
    data = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(src_texts, hypotheses, references)]
    result = _COMET_MODEL.predict(data, progress_bar=False)
    return {"comet": round(float(result.system_score), 4)}


def evaluate(reference_srt: str, hypothesis_srt: str) -> dict:
    ref_items = parse_srt(reference_srt)
    hyp_items = parse_srt(hypothesis_srt)
    ref_map = {item.id: item for item in ref_items}
    hyp_map = {item.id: item for item in hyp_items}
    common_ids = sorted(set(ref_map) & set(hyp_map))
    if not common_ids:
        return {"error": "no matching subtitle IDs found"}

    ref_texts = [ref_map[i].text for i in common_ids]
    hyp_texts = [hyp_map[i].text for i in common_ids]

    translation: dict = {}
    if len(ref_texts) >= 1:
        translation.update(sacrebleu_score(ref_texts, hyp_texts))
        translation.update(chrf_score(ref_texts, hyp_texts))
        translation.update(comet_score(ref_texts, hyp_texts, src_texts=ref_texts))

    sub_metrics = SubtitleMetrics()
    subtitle = sub_metrics.evaluate([ref_map[i] for i in common_ids], hyp_texts)

    return {"translation": translation, "subtitle": subtitle, "num_items": len(common_ids)}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate subtitle translations")
    parser.add_argument("--reference", type=str, required=True, help="Reference SRT file")
    parser.add_argument("--hypothesis", type=str, required=True, help="Hypothesis SRT file")
    parser.add_argument("--output", type=str, default="", help="JSON output path")
    args = parser.parse_args()
    results = evaluate(args.reference, args.hypothesis)
    output = json.dumps(results, indent=2)
    if args.output:
        Path(args.output).write_text(output)
    print(output)
