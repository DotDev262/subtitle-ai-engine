from __future__ import annotations

from subtitle_nmt_thesis.data.parser import SubtitleItem


class ContextBuilder:
    def __init__(self, window_size: int = 2) -> None:
        self.window_size = window_size

    def build(
        self,
        items: list[SubtitleItem],
        idx: int,
        course: str = "",
        topic: str = "",
        glossary: dict[str, str] | None = None,
    ) -> dict:
        prev_start = max(0, idx - self.window_size)
        previous = items[prev_start:idx]
        next_end = min(len(items), idx + self.window_size + 1)
        nxt = items[idx + 1 : next_end]
        return {
            "previous": previous,
            "current": items[idx],
            "next": nxt,
            "course": course,
            "topic": topic,
            "glossary": glossary or {},
        }

    def format_context(self, ctx: dict) -> str:
        parts: list[str] = []
        if ctx.get("previous"):
            prev_text = " | ".join(s.text for s in ctx["previous"])
            parts.append(f"[Previous: {prev_text}]")
        if ctx.get("next"):
            next_text = " | ".join(s.text for s in ctx["next"])
            parts.append(f"[Next: {next_text}]")
        if ctx.get("course"):
            parts.append(f"[Course: {ctx['course']}]")
        if ctx.get("topic"):
            parts.append(f"[Topic: {ctx['topic']}]")
        if ctx.get("glossary"):
            gloss = "; ".join(f"{k} -> {v}" for k, v in ctx["glossary"].items())
            parts.append(f"[Glossary: {gloss}]")
        return " ".join(parts) + " " if parts else ""
