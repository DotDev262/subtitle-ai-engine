import json
from pathlib import Path

import sacrebleu

from subtitle_nmt_thesis.data.parser import parse_srt
from subtitle_nmt_thesis.evaluation.subtitle_metrics import SubtitleMetrics


def sacrebleu_score(references: list[str], hypotheses: list[str]) -> dict:
    refs = [[r] for r in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    return {"bleu": bleu.score, "precisions": bleu.precisions}


def chrf_score(references: list[str], hypotheses: list[str]) -> dict:
    refs = [[r] for r in references]
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    return {"chrf": chrf.score}


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

    translation = {}
    if len(ref_texts) >= 1:
        translation.update(sacrebleu_score(ref_texts, hyp_texts))
        translation.update(chrf_score(ref_texts, hyp_texts))

    sub_metrics = SubtitleMetrics()
    subtitle = sub_metrics.evaluate([ref_map[i] for i in common_ids], hyp_texts)

    return {"translation": translation, "subtitle": subtitle, "num_items": len(common_ids)}


def main():
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
