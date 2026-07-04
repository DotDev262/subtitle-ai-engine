from subtitle_nmt_thesis.data.parser import SubtitleItem


class CPSConstraint:
    def __init__(self, max_cps: int = 21):
        self.max_cps = max_cps

    def score(self, text: str, item: SubtitleItem | None = None) -> float:
        if item is None or item.end == item.start:
            return 0.0
        duration = item.end - item.start
        cps = len(text) / duration
        if cps <= self.max_cps:
            return 0.0
        return (cps - self.max_cps) / self.max_cps
