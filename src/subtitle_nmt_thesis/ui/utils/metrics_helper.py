from subtitle_nmt_thesis.data.parser import SubtitleItem


def compute_cue_metrics(
    text: str, duration_sec: float, max_cpl: int = 42, max_cps: int = 21
) -> dict:
    lines = text.splitlines() if text else [""]
    max_line_cpl = max((len(line) for line in lines), default=0)
    char_count = len(text)
    dur = max(duration_sec, 0.1)
    cps = char_count / dur
    return {
        "char_count": char_count,
        "line_count": len(lines),
        "max_line_cpl": max_line_cpl,
        "cps": round(cps, 1),
        "cpl_compliant": max_line_cpl <= max_cpl,
        "cps_compliant": cps <= max_cps,
    }


def compute_aggregate_metrics(
    source_items: list[SubtitleItem],
    target_items: list[SubtitleItem],
    max_cpl: int = 42,
    max_cps: int = 21,
) -> dict:
    if not target_items:
        return {
            "total_cues": 0,
            "cpl_violation_rate": 0.0,
            "cps_violation_rate": 0.0,
            "avg_cpl": 0.0,
            "avg_cps": 0.0,
            "length_ratio": 1.0,
        }
    total_lines = 0
    cpl_violations = 0
    cps_violations = 0
    total_cpl = 0
    total_cps = 0
    total_src_chars = sum(len(it.text) for it in source_items)
    total_tgt_chars = sum(len(it.text) for it in target_items)

    for it in target_items:
        duration = max(it.end - it.start, 0.1)
        lines = it.text.splitlines() or [""]
        total_lines += len(lines)
        for line in lines:
            total_cpl += len(line)
            if len(line) > max_cpl:
                cpl_violations += 1
        cps = len(it.text) / duration
        total_cps += cps
        if cps > max_cps:
            cps_violations += 1

    cpl_rate = (cpl_violations / total_lines * 100.0) if total_lines else 0.0
    cps_rate = (cps_violations / len(target_items) * 100.0) if target_items else 0.0
    avg_cpl = total_cpl / total_lines if total_lines else 0.0
    avg_cps = total_cps / len(target_items) if target_items else 0.0
    ratio = (total_tgt_chars / total_src_chars) if total_src_chars else 1.0

    return {
        "total_cues": len(target_items),
        "total_lines": total_lines,
        "cpl_violation_rate": round(cpl_rate, 1),
        "cps_violation_rate": round(cps_rate, 1),
        "avg_cpl": round(avg_cpl, 1),
        "avg_cps": round(avg_cps, 1),
        "length_ratio": round(ratio, 2),
    }
