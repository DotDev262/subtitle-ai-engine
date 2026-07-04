class ReadabilityConstraint:
    # ponytail: avg word length only; no syllable or sentence-length heuristics
    def score(self, text: str, item=None) -> float:
        words = text.split()
        if not words:
            return 0.0
        avg_word_len = sum(len(w) for w in words) / len(words)
        return min(avg_word_len / 15.0, 1.0)
