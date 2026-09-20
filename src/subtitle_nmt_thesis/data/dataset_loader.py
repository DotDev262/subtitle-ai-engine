import json
from pathlib import Path
from datasets import load_dataset
from subtitle_nmt_thesis.data.parser import SubtitleItem, write_srt


def _parse_nptel_stream(stream, num_samples: int) -> list[dict[str, str]]:
    samples = []
    for item in stream:
        src = item.get("sentence", "") or item.get("source", "")
        tgt = item.get("translation", "") or item.get("target", "")
        if src and tgt:
            samples.append({"source": src.strip(), "target": tgt.strip()})
        if len(samples) >= num_samples:
            break
    return samples


def _parse_iitb_stream(stream, num_samples: int) -> list[dict[str, str]]:
    samples = []
    for item in stream:
        trans = item.get("translation", {})
        src = trans.get("en", "").strip()
        tgt = trans.get("hi", "").strip()
        if src and tgt:
            samples.append({"source": src, "target": tgt})
        if len(samples) >= num_samples:
            break
    return samples


def load_parallel_samples(
    dataset_name: str = "nptel",
    split: str = "test",
    num_samples: int = 100,
    streaming: bool = True,
) -> list[dict[str, str]]:
    """Loads parallel samples from NPTEL or IITB with graceful fallback."""
    name_lower = dataset_name.lower()
    if "nptel" in name_lower:
        try:
            # Attempt to stream from NPTEL
            ds = load_dataset("ai4bharat/NPTEL", "en2indic", split=split, streaming=streaming)
            return _parse_nptel_stream(ds, num_samples)
        except Exception:
            # Graceful fallback to IITB if NPTEL is gated/unauthenticated
            return load_parallel_samples(
                "iitb", split="test", num_samples=num_samples, streaming=streaming
            )
    else:
        ds = load_dataset("cfilt/iitb-english-hindi", split=split, streaming=streaming)
        return _parse_iitb_stream(ds, num_samples)


def export_to_jsonl(samples: list[dict[str, str]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def export_to_srt(
    samples: list[dict[str, str]],
    output_src_srt: str | Path,
    output_tgt_srt: str | Path,
    duration_per_cue: float = 3.5,
) -> None:
    src_items = []
    tgt_items = []
    current_time = 0.0

    for idx, s in enumerate(samples, start=1):
        start = current_time
        end = current_time + duration_per_cue
        src_items.append(SubtitleItem(id=idx, start=start, end=end, text=s["source"]))
        tgt_items.append(SubtitleItem(id=idx, start=start, end=end, text=s["target"]))
        current_time = end + 0.5

    write_srt(src_items, output_src_srt)
    write_srt(tgt_items, output_tgt_srt)
