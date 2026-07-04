from pathlib import Path
import csv


def write_human_eval_csv(
    items: list, source_texts: list[str], translations: list[str], output_path: str | Path
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "source", "translation", "fluency", "readability", "accuracy", "comprehension"])
        for item, src, trans in zip(items, source_texts, translations):
            writer.writerow([item.id, src, trans, "", "", "", ""])
