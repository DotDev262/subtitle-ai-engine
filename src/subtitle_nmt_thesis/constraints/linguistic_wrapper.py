import re

INDIC_BREAK_MARKERS = {
    # Conjunctions & transitions
    "और",
    "तथा",
    "एवं",
    "या",
    "अथवा",
    "लेकिन",
    "किंतु",
    "परंतु",
    "मगर",
    "क्योंकि",
    "ताकि",
    "जिससे",
    # Postpositions that form natural phrase boundaries
    "के",
    "की",
    "का",
    "में",
    "पर",
    "से",
    "को",
    "द्वारा",
    # English equivalents
    "and",
    "or",
    "but",
    "because",
    "so",
    "that",
    "with",
    "from",
    "into",
    "onto",
}


def wrap_subtitles_syntactic(text: str, max_cpl: int = 42, max_lines: int = 2) -> str:
    """
    Splits a subtitle cue into grammatically coherent lines <= max_cpl,
    preferring clause and conjunction boundaries over greedy splits.
    """
    clean_text = re.sub(r"\s+", " ", text).strip()
    if not clean_text or len(clean_text) <= max_cpl:
        return clean_text

    words = clean_text.split(" ")
    total_len = len(clean_text)
    target_split = total_len // max_lines
    best_split_idx = -1
    best_score = float("inf")

    running_len = 0
    for idx, word in enumerate(words[:-1]):
        running_len += len(word) + (1 if idx > 0 else 0)
        line1 = " ".join(words[: idx + 1])
        line2 = " ".join(words[idx + 1 :])

        if len(line1) > max_cpl or len(line2) > max_cpl:
            continue

        balance_penalty = abs(len(line1) - target_split)
        next_word = words[idx + 1].strip(".,|।?!")
        current_last = word.strip()

        syntactic_bonus = 0
        if current_last.endswith(("।", ".", ",", ";", "?", "!")):
            syntactic_bonus = -20
        elif next_word in INDIC_BREAK_MARKERS:
            syntactic_bonus = -15
        elif current_last in INDIC_BREAK_MARKERS:
            syntactic_bonus = -10

        score = balance_penalty + syntactic_bonus
        if score < best_score:
            best_score = score
            best_split_idx = idx

    if best_split_idx != -1:
        line1 = " ".join(words[: best_split_idx + 1])
        line2 = " ".join(words[best_split_idx + 1 :])
        return f"{line1}\n{line2}"

    # Fallback: standard greedy word-level wrap
    lines = []
    curr = []
    curr_len = 0
    for w in words:
        space = 1 if curr else 0
        if curr_len + len(w) + space > max_cpl and curr:
            lines.append(" ".join(curr))
            curr = [w]
            curr_len = len(w)
        else:
            curr.append(w)
            curr_len += len(w) + space
    if curr:
        lines.append(" ".join(curr))
    return "\n".join(lines[:max_lines]) if max_lines else "\n".join(lines)
