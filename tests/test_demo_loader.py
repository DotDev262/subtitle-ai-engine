from pathlib import Path
from subtitle_nmt_thesis.ui.utils.demo_loader import get_demo_asset_paths, load_demo_srt


def test_demo_assets_exist():
    assets = get_demo_asset_paths()
    assert "video" in assets
    assert "srt_en" in assets
    assert "srt_hi" in assets
    assert assets["video"].exists()
    assert assets["srt_en"].exists()
    assert assets["srt_hi"].exists()


def test_load_demo_srt():
    content = load_demo_srt("en")
    assert "00:00:01,000 --> 00:00:04,000" in content
    assert len(content) > 0
