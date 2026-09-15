from subtitle_nmt_thesis.ui.components.studio import prepare_subtitles_for_studio
from subtitle_nmt_thesis.data.parser import SubtitleItem


def test_prepare_subtitles_for_studio():
    items = [
        SubtitleItem(id=1, start=1.0, end=4.0, text="Line 1"),
        SubtitleItem(id=2, start=5.0, end=8.0, text="Line 2"),
    ]
    srt_out, vtt_out = prepare_subtitles_for_studio(items)
    assert "00:00:01,000 --> 00:00:04,000" in srt_out
    assert "WEBVTT" in vtt_out
    assert "00:00:01.000 --> 00:00:04.000" in vtt_out
