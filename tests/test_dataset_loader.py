from pathlib import Path
from subtitle_nmt_thesis.data.dataset_loader import (
    export_to_jsonl,
    export_to_srt,
)


def test_export_to_jsonl(tmp_path: Path):
    samples = [{"source": "Hello", "target": "नमस्ते"}]
    out_file = tmp_path / "test.jsonl"
    export_to_jsonl(samples, out_file)
    assert out_file.exists()
    assert '"source": "Hello"' in out_file.read_text(encoding="utf-8")


def test_export_to_srt(tmp_path: Path):
    samples = [{"source": "Lecture 1 intro", "target": "व्याख्यान 1 परिचय"}]
    src_srt = tmp_path / "src.srt"
    tgt_srt = tmp_path / "tgt.srt"
    export_to_srt(samples, src_srt, tgt_srt, duration_per_cue=4.0)
    assert src_srt.exists() and tgt_srt.exists()
    assert "00:00:00,000 --> 00:00:04,000" in src_srt.read_text(encoding="utf-8")
    assert "Lecture 1 intro" in src_srt.read_text(encoding="utf-8")
    assert "व्याख्यान 1 परिचय" in tgt_srt.read_text(encoding="utf-8")
