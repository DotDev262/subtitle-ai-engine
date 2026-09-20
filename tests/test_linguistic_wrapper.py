from subtitle_nmt_thesis.constraints.linguistic_wrapper import wrap_subtitles_syntactic


def test_wrap_single_short_line():
    text = "आज हम ट्री ट्रैवर्सल समझेंगे।"
    wrapped = wrap_subtitles_syntactic(text, max_cpl=42)
    assert wrapped == text
    assert len(wrapped.splitlines()) == 1


def test_wrap_splits_at_conjunction_boundary():
    text = "आज हम संतुलित बाइनरी सर्च ट्री और उनकी समय जटिलता समझेंगे।"
    wrapped = wrap_subtitles_syntactic(text, max_cpl=42, max_lines=2)
    lines = wrapped.splitlines()
    assert len(lines) <= 2
    for line in lines:
        assert len(line) <= 42
    assert any(line.strip().startswith("और") or line.strip().endswith("ट्री") for line in lines)


def test_wrap_handles_no_overflow_without_word_slicing():
    text = "यह एक बहुत लंबा वाक्य है जो कई शब्दों से मिलकर बना है ताकि सीमा का परीक्षण हो सके।"
    wrapped = wrap_subtitles_syntactic(text, max_cpl=40, max_lines=2)
    lines = wrapped.splitlines()
    for line in lines:
        assert len(line) <= 40
