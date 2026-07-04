from pathlib import Path
import tempfile
from subtitle_nmt_thesis.data.parser import SubtitleItem, parse_srt, write_srt

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:04,000
Hello, welcome to the course.

2
00:00:05,000 --> 00:00:08,000
Today we will cover binary trees.
"""


def test_parse_srt():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
        f.write(SAMPLE_SRT)
        f.flush()
        items = parse_srt(f.name)
    assert len(items) == 2
    assert items[0].id == 1
    assert items[0].start == 1.0
    assert items[0].end == 4.0
    assert items[0].text == "Hello, welcome to the course."
    assert items[1].id == 2
    assert items[1].text == "Today we will cover binary trees."


def test_write_srt():
    items = [
        SubtitleItem(id=1, start=1.0, end=4.0, text="Hello."),
        SubtitleItem(id=2, start=5.0, end=8.0, text="World."),
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False) as f:
        f.write("")
    write_srt(items, f.name)
    written = Path(f.name).read_text()
    assert "Hello." in written
    assert "World." in written
    assert "00:00:01,000" in written
    assert "00:00:08,000" in written
    Path(f.name).unlink()
