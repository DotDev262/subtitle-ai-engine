"""Subtitle serialization helpers used by the React backend."""

import tempfile
from pathlib import Path

from subtitle_nmt_thesis.data.parser import SubtitleItem, write_srt
from subtitle_nmt_thesis.ui.utils.srt_vtt import srt_to_vtt


def prepare_subtitles_for_studio(items: list[SubtitleItem]) -> tuple[str, str]:
    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        write_srt(items, path)
        srt_content = path.read_text(encoding="utf-8")
    finally:
        path.unlink(missing_ok=True)
    return srt_content, srt_to_vtt(srt_content)
