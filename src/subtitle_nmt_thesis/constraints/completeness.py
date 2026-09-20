import re

HANGING_MARKERS = {
    "और",
    "तथा",
    "या",
    "एवं",
    "किंतु",
    "परंतु",
    "का",
    "के",
    "की",
    "में",
    "पर",
    "से",
    "को",
    "and",
    "or",
    "but",
    "the",
    "a",
    "an",
    "of",
    "in",
    "to",
    "for",
    "with",
    "at",
}


class CompletenessConstraint:
    def __init__(self, penalty: float = 2.5):
        self.penalty = penalty

    def score(self, text: str, item=None) -> float:
        cleaned = text.strip()
        if not cleaned:
            return self.penalty

        score = 0.0
        # Check terminal punctuation
        if not cleaned.endswith(("।", ".", "?", "!", '"', "”", "'")):
            score += 1.0

        # Check for hanging words at end
        words = cleaned.split()
        last_word = re.sub(r"[^\w]", "", words[-1]).lower() if words else ""
        if last_word in HANGING_MARKERS:
            score += 1.5

        # Check for single character trailing token (typical artifact of cutoffs)
        if len(last_word) == 1 and not cleaned.endswith(("।", ".", "?", "!")):
            score += 2.0

        return score * self.penalty
