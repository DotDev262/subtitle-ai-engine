import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubtitleItem:
    id: int
    start: float
    end: float
    text: str


def _ts_to_seconds(ts: str) -> float:
    h, m, s = ts.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _seconds_to_ts(total: float) -> str:
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = total - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def parse_srt(path: str | Path) -> list[SubtitleItem]:
    text = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", text.strip())
    items = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        idx = int(lines[0])
        times = lines[1].split(" --> ")
        start = _ts_to_seconds(times[0])
        end = _ts_to_seconds(times[1])
        content = " ".join(lines[2:])
        items.append(SubtitleItem(id=idx, start=start, end=end, text=content))
    return items


def write_srt(items: list[SubtitleItem], path: str | Path) -> None:
    lines = []
    for item in items:
        start = _seconds_to_ts(item.start)
        end = _seconds_to_ts(item.end)
        lines.append(str(item.id))
        lines.append(f"{start} --> {end}")
        lines.append(item.text)
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
