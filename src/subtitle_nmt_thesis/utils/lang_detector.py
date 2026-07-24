from __future__ import annotations
from pathlib import Path
from langdetect import detect_langs, DetectorFactory

from subtitle_nmt_thesis.data.parser import SubtitleItem

DetectorFactory.seed = 0


def detect_language(input_path: str | Path, sample_size: int = 10) -> str:
    items = _parse_sample(input_path, sample_size)
    text = " ".join(item.text for item in items if item.text.strip())
    if not text:
        return "en"
    langs: list = detect_langs(text)
    top = langs[0]
    code = top.lang
    if code in ("hi", "bn", "ta", "te"):
        return code
    return code


def _parse_sample(path: str | Path, n: int) -> list[SubtitleItem]:
    from subtitle_nmt_thesis.data.parser import parse_srt
    items = parse_srt(path)
    step = max(1, len(items) // n)
    return items[::step][:n]
