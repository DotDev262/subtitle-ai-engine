class CPLConstraint:
    def __init__(self, max_cpl: int = 42):
        self.max_cpl = max_cpl

    def score(self, text: str, item=None) -> float:
        total = 0.0
        for line in text.splitlines():
            if len(line) > self.max_cpl:
                total += (len(line) - self.max_cpl) / self.max_cpl
        return total
