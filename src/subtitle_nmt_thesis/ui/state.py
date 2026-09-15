import torch
import streamlit as st
from subtitle_nmt_thesis.pipeline.run import _build_model, _auto_resolve_model
from subtitle_nmt_thesis.ui.utils.metrics_helper import compute_cue_metrics


def get_device_status() -> tuple[str, bool]:
    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else "cpu"
    return device, is_cuda


@st.cache_resource(show_spinner="Loading NMT model into memory...")
def get_cached_translator(model_name: str, src_lang: str, tgt_lang: str):
    resolved = _auto_resolve_model(model_name, src_lang, tgt_lang)
    return _build_model(resolved, src_lang, tgt_lang)


def run_single_comparison(
    text: str,
    model,
    max_cpl: int = 42,
    max_cps: int = 21,
    duration_sec: float = 3.0,
) -> dict:
    baseline_out = model.translate(text)
    if hasattr(model, "translate_constrained"):
        constrained_out = model.translate_constrained(text, max_cpl=max_cpl)
    else:
        constrained_out = baseline_out

    base_metrics = compute_cue_metrics(baseline_out, duration_sec, max_cpl, max_cps)
    const_metrics = compute_cue_metrics(constrained_out, duration_sec, max_cpl, max_cps)

    return {
        "baseline_text": baseline_out,
        "baseline_metrics": base_metrics,
        "constrained_text": constrained_out,
        "constrained_metrics": const_metrics,
    }
