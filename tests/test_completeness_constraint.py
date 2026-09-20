from subtitle_nmt_thesis.constraints.completeness import CompletenessConstraint


def test_complete_sentence_has_zero_penalty():
    c = CompletenessConstraint()
    assert c.score("आज हम बाइनरी ट्री समझेंगे।") == 0.0
    assert c.score("We will explore algorithms today.") == 0.0


def test_truncated_sentence_has_high_penalty():
    c = CompletenessConstraint()
    # Sentence abruptly truncated mid-sentence or mid-word
    assert c.score("आज हम संतुलित बाइनरी खोज वृक्षों और उनके अ") > 0.0
    assert c.score("Today we will analyze balanced binary search and") > 0.0
