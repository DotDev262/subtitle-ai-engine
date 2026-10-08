"""Pure compatibility helpers for subtitle comparisons.

The React frontend owns presentation state. These functions remain available
for the Python test suite and for callers that need comparison calculations.
"""

from subtitle_nmt_thesis.constraints.completeness import CompletenessConstraint
from subtitle_nmt_thesis.constraints.linguistic_wrapper import wrap_subtitles_syntactic
from subtitle_nmt_thesis.ui.utils.metrics_helper import compute_cue_metrics


def get_device_status() -> tuple[str, bool]:
    try:
        import torch
    except ImportError:
        return "cpu", False

    is_cuda = torch.cuda.is_available()
    return ("cuda" if is_cuda else "cpu"), is_cuda


def wrap_subtitles_to_cpl(text: str, max_cpl: int = 42) -> str:
    """Wrap text at word boundaries without exceeding the target line width."""
    words = text.split()
    if not words:
        return ""

    lines: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        separator_length = 1 if current else 0
        if current and current_length + len(word) + separator_length > max_cpl:
            lines.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += len(word) + separator_length

    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def run_single_comparison(
    text: str,
    model,
    max_cpl: int = 42,
    max_cps: int = 21,
    duration_sec: float = 3.0,
) -> dict:
    """Compare baseline output with the best constraint-aware candidate."""
    baseline_text = model.translate(text)
    candidates = [baseline_text]

    if hasattr(model, "translate_n"):
        candidates = model.translate_n(text, n=5)

    if hasattr(model, "translate_constrained"):
        constrained_text = model.translate_constrained(text, max_cpl=max_cpl)
        if constrained_text not in candidates:
            candidates.append(constrained_text)

    completeness_scorer = CompletenessConstraint(penalty=3.0)
    duration = max(duration_sec, 0.1)
    best_text = candidates[0]
    best_score = float("inf")

    for candidate in candidates:
        formatted = wrap_subtitles_syntactic(candidate, max_cpl=max_cpl, max_lines=2)
        lines = formatted.splitlines()
        max_line = max((len(line) for line in lines), default=0)
        cps = len(candidate) / duration
        score = (
            max(0, max_line - max_cpl) * 3.0
            + max(0, cps - max_cps) * 2.0
            + completeness_scorer.score(candidate)
            + max(0, len(lines) - 2) * 5.0
            + len(candidate) * 0.02
        )
        if score < best_score:
            best_score = score
            best_text = formatted

    return {
        "baseline_text": baseline_text,
        "baseline_metrics": compute_cue_metrics(
            baseline_text, duration_sec, max_cpl, max_cps
        ),
        "constrained_text": best_text,
        "constrained_metrics": compute_cue_metrics(
            best_text, duration_sec, max_cpl, max_cps
        ),
    }
