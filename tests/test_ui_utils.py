from subtitle_nmt_thesis.ui.utils.srt_vtt import srt_to_vtt
from subtitle_nmt_thesis.ui.utils.metrics_helper import compute_cue_metrics, compute_aggregate_metrics
from subtitle_nmt_thesis.data.parser import SubtitleItem

def test_srt_to_vtt_conversion():
    srt_content = (
        "1\n"
        "00:00:01,500 --> 00:00:04,250\n"
        "Hello world.\n\n"
        "2\n"
        "00:00:05,000 --> 00:00:07,100\n"
        "Welcome to the lecture.\n"
    )
    vtt_content = srt_to_vtt(srt_content)
    assert vtt_content.startswith("WEBVTT")
    assert "00:00:01.500 --> 00:00:04.250" in vtt_content
    assert "00:00:05.000 --> 00:00:07.100" in vtt_content
    assert "Hello world." in vtt_content

def test_compute_cue_metrics():
    text = "Short line\nAnother line"
    metrics = compute_cue_metrics(text=text, duration_sec=2.0, max_cpl=42, max_cps=21)
    assert metrics["char_count"] == len(text)
    assert metrics["max_line_cpl"] == len("Another line")
    assert metrics["cpl_compliant"] is True
    assert metrics["cps_compliant"] is True

def test_compute_cue_metrics_violation():
    long_text = "A" * 50
    metrics = compute_cue_metrics(text=long_text, duration_sec=1.0, max_cpl=42, max_cps=21)
    assert metrics["cpl_compliant"] is False
    assert metrics["cps_compliant"] is False

def test_compute_aggregate_metrics():
    src = [SubtitleItem(id=1, start=1.0, end=3.0, text="Hello world")]
    tgt = [SubtitleItem(id=1, start=1.0, end=3.0, text="नमस्ते दुनिया")]
    agg = compute_aggregate_metrics(src, tgt, max_cpl=42, max_cps=21)
    assert "cpl_violation_rate" in agg
    assert "cps_violation_rate" in agg
    assert agg["cpl_violation_rate"] == 0.0
