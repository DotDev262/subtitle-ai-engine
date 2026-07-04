import re


class TerminologyConstraint:
    def __init__(self, glossary: dict[str, str] | None = None, penalty: float = 1.0):
        self.glossary = glossary or {}
        self.penalty = penalty

    def score(self, text: str, item=None) -> float:
        lower = text.lower()
        total = 0.0
        for term, preferred in self.glossary.items():
            if re.search(r"\b" + re.escape(term.lower()) + r"\b", lower):
                continue
            if preferred and re.search(r"\b" + re.escape(preferred.lower()) + r"\b", lower):
                continue
            total += self.penalty
        return total
