from subtitle_nmt_thesis.evaluation.subtitle_metrics import SubtitleMetrics
from subtitle_nmt_thesis.data.parser import SubtitleItem


def test_cpl_violations():
    items = [SubtitleItem(id=1, start=0.0, end=2.0, text="")]
    translations = ["x" * 100]
    m = SubtitleMetrics(max_cpl=42)
    result = m.evaluate(items, translations)
    assert result["cpl_violations"] > 0


def test_cps_violations():
    items = [SubtitleItem(id=1, start=0.0, end=1.0, text="")]
    translations = ["x" * 100]
    m = SubtitleMetrics(max_cps=21)
    result = m.evaluate(items, translations)
    assert result["cps_violations"] > 0


def test_perfect_subtitles():
    items = [SubtitleItem(id=1, start=0.0, end=10.0, text="")]
    translations = ["short"]
    m = SubtitleMetrics(max_cpl=42, max_cps=21)
    result = m.evaluate(items, translations)
    assert result["cpl_violations"] == 0
    assert result["cps_violations"] == 0
