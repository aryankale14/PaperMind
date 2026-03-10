def section_score(text: str) -> float:
    """
    Assigns importance score based on academic section clues.
    """

    text_lower = text.lower()

    score = 1.0

    # High-value signals
    if "introduction" in text_lower:
        score += 2

    if "this study" in text_lower or "this paper" in text_lower:
        score += 2

    if "we propose" in text_lower or "we present" in text_lower:
        score += 2

    if "research question" in text_lower:
        score += 3

    # Penalize references
    if "vol." in text_lower and "pp." in text_lower:
        score -= 2

    if text_lower.count(",") > 15:
        score -= 1.5

    return score