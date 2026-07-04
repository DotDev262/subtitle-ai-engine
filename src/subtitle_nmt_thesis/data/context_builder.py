from subtitle_nmt_thesis.data.parser import SubtitleItem


class ContextBuilder:
    def __init__(self, window_size: int = 2):
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
