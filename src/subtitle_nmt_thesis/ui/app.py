"""Compatibility helpers for callers that construct the Python pipeline."""

from subtitle_nmt_thesis.pipeline.run import Pipeline


def build_pipeline_instance(
    max_cpl: int = 42,
    max_cps: int = 21,
    constrained_decoding: bool = True,
    src_lang: str = "en",
    tgt_lang: str = "hi",
    model_name: str = "auto",
) -> Pipeline:
    return Pipeline(
        model_name=model_name,
        max_cpl=max_cpl,
        max_cps=max_cps,
        constrained_decoding=constrained_decoding,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )
