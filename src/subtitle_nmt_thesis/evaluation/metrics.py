import sacrebleu


def sacrebleu_score(references: list[str], hypotheses: list[str]) -> dict:
    refs = [[r] for r in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    return {"bleu": bleu.score, "precisions": bleu.precisions}


def chrf_score(references: list[str], hypotheses: list[str]) -> dict:
    refs = [[r] for r in references]
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    return {"chrf": chrf.score}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate subtitle translations")
    parser.add_argument("--reference", type=str, required=True, help="Reference SRT file")
    parser.add_argument("--hypothesis", type=str, required=True, help="Hypothesis SRT file")
    args = parser.parse_args()
    print(f"Evaluating {args.hypothesis} against {args.reference}")
