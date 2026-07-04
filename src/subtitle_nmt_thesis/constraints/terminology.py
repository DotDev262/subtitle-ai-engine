class TerminologyConstraint:
    def __init__(self, glossary: dict[str, str] | None = None, penalty: float = 1.0):
        self.glossary = glossary or {}
        self.penalty = penalty

    def score(self, text: str, item=None) -> float:
        lower = text.lower()
        total = 0.0
        for term, preferred in self.glossary.items():
            if term.lower() in lower:
                continue
            if preferred.lower() in lower:
                continue
            for word in term.lower().split():
                if word in lower:
                    total += self.penalty
                    break
        return total
