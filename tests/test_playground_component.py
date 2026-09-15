from subtitle_nmt_thesis.ui.components.playground import PRESETS


def test_playground_presets():
    assert len(PRESETS) >= 3
    for p in PRESETS:
        assert "text" in p
        assert "duration" in p
        assert len(p["text"]) > 0
