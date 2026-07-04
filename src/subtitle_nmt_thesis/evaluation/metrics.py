import sacrebleu


def sacrebleu_score(references: list[str], hypotheses: list[str]) -> dict:
    refs = [[r] for r in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    return {"bleu": bleu.score, "precisions": bleu.precisions}


def chrf_score(references: list[str], hypotheses: list[str]) -> dict:
    refs = [[r] for r in references]
    chrf = sacrebleu.corpus_chrf(hypotheses, refs)
    return {"chrf": chrf.score}
