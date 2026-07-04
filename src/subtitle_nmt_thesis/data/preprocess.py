import re
from subtitle_nmt_thesis.data.parser import SubtitleItem


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_long_subtitle(item: SubtitleItem, max_cpl: int = 42) -> list[SubtitleItem]:
    if len(item.text) <= max_cpl:
        return [item]
    words = item.text.split()
    chunks = []
    current = []
    current_len = 0
    for word in words:
        if len(word) > max_cpl:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            for i in range(0, len(word), max_cpl):
                chunks.append(word[i:i+max_cpl])
        elif current_len + len(word) + 1 > max_cpl and current:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    duration = item.end - item.start
    per_chunk = duration / len(chunks) if chunks else duration
    return [
        SubtitleItem(id=item.id, start=item.start + i * per_chunk, end=item.start + (i + 1) * per_chunk, text=chunk)
        for i, chunk in enumerate(chunks)
    ]


def main():
    import argparse
    from subtitle_nmt_thesis.data.parser import parse_srt, write_srt

    parser = argparse.ArgumentParser(description="Preprocess subtitle files")
    parser.add_argument("input", type=str, help="Input SRT file")
    parser.add_argument("--output", type=str, default="output.srt", help="Output SRT path")
    parser.add_argument("--max-cpl", type=int, default=42, help="Max characters per line")
    args = parser.parse_args()
    items = parse_srt(args.input)
    cleaned = []
    for item in items:
        item.text = clean_text(item.text)
        cleaned.extend(split_long_subtitle(item, max_cpl=args.max_cpl))
    write_srt(cleaned, args.output)
    print(f"Preprocessed {args.input} -> {args.output}: {len(items)} -> {len(cleaned)} subtitles")
