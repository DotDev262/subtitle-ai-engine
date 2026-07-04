from subtitle_nmt_thesis.data.parser import SubtitleItem


class SubtitleMetrics:
    def __init__(self, max_cpl: int = 42, max_cps: int = 21):
        self.max_cpl = max_cpl
        self.max_cps = max_cps

    def evaluate(self, items: list[SubtitleItem], translations: list[str]) -> dict:
        cpl_violations = 0
        cps_violations = 0
        for item, trans in zip(items, translations):
            for line in trans.splitlines():
                if len(line) > self.max_cpl:
                    cpl_violations += 1
            duration = item.end - item.start
            if duration > 0 and len(trans) / duration > self.max_cps:
                cps_violations += 1
        return {
            "cpl_violations": cpl_violations,
            "cps_violations": cps_violations,
            "total_items": len(items),
        }
