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


def wrap_subtitles_to_cpl(text: str, max_cpl: int = 42) -> str:
    """Format subtitle text across lines (max 2 lines) without breaking words."""
    words = text.split()
    if not words:
        return ""
    lines = []
    current = []
    current_len = 0
    for w in words:
        space = 1 if current else 0
        if current_len + len(w) + space > max_cpl and current:
            lines.append(" ".join(current))
            current = [w]
            current_len = len(w)
        else:
            current.append(w)
            current_len += len(w) + space
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


from subtitle_nmt_thesis.constraints.linguistic_wrapper import wrap_subtitles_syntactic
from subtitle_nmt_thesis.constraints.completeness import CompletenessConstraint

completeness_scorer = CompletenessConstraint(penalty=3.0)


def run_single_comparison(
    text: str,
    model,
    max_cpl: int = 42,
    max_cps: int = 21,
    duration_sec: float = 3.0,
) -> dict:
    # Baseline: standard single unconstrained generation
    baseline_raw = model.translate(text)
    baseline_out = baseline_raw

    # Constraint-aware: combine candidate generation, reranking and syntactic line wrapping
    if hasattr(model, "translate_n"):
        try:
            candidates = model.translate_n(text, n=5)
        except Exception:
            candidates = [baseline_raw]
    else:
        candidates = [baseline_raw]

    if hasattr(model, "translate_constrained"):
        try:
            direct_constrained = model.translate_constrained(text, max_cpl=max_cpl)
            if direct_constrained not in candidates:
                candidates.append(direct_constrained)
        except Exception:
            pass

    # Score each candidate against CPL, CPS and Completeness criteria
    best_text = candidates[0]
    best_score = float("inf")
    dur = max(duration_sec, 0.1)

    for cand in candidates:
        formatted_cand = wrap_subtitles_syntactic(cand, max_cpl=max_cpl, max_lines=2)
        lines = formatted_cand.splitlines()
        max_line = max((len(l) for l in lines), default=0)
        cps = len(cand) / dur
        
        cpl_penalty = max(0, max_line - max_cpl) * 3.0
        cps_penalty = max(0, cps - max_cps) * 2.0
        complete_penalty = completeness_scorer.score(cand)
        
        # Penalize line counts exceeding standard 2 lines
        line_count_penalty = max(0, len(lines) - 2) * 5.0
        
        score = cpl_penalty + cps_penalty + complete_penalty + line_count_penalty + (len(cand) * 0.02)
        if score < best_score:
            best_score = score
            best_text = formatted_cand

    constrained_out = best_text

    base_metrics = compute_cue_metrics(baseline_out, duration_sec, max_cpl, max_cps)
    const_metrics = compute_cue_metrics(constrained_out, duration_sec, max_cpl, max_cps)

    return {
        "baseline_text": baseline_out,
        "baseline_metrics": base_metrics,
        "constrained_text": constrained_out,
        "constrained_metrics": const_metrics,
    }
