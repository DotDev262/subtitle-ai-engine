import pytest
from subtitle_nmt_thesis.constraints.cpl import CPLConstraint
from subtitle_nmt_thesis.constraints.cps import CPSConstraint
from subtitle_nmt_thesis.constraints.readability import ReadabilityConstraint
from subtitle_nmt_thesis.constraints.terminology import TerminologyConstraint
from subtitle_nmt_thesis.data.parser import SubtitleItem


def test_cpl_short_line():
    c = CPLConstraint(max_cpl=42)
    assert c.score("short") == 0.0


def test_cpl_exceeded():
    c = CPLConstraint(max_cpl=10)
    assert c.score("x" * 20) > 0.0


def test_cps_within_limit():
    item = SubtitleItem(id=1, start=0.0, end=10.0, text="short text")
    c = CPSConstraint(max_cps=21)
    assert c.score("short text", item=item) == 0.0


def test_cps_exceeded():
    item = SubtitleItem(id=1, start=0.0, end=1.0, text="x" * 50)
    c = CPSConstraint(max_cps=21)
    assert c.score("x" * 50, item=item) > 0.0


def test_readability_simple():
    c = ReadabilityConstraint()
    score = c.score("Hello world.")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_terminology_forced_choice():
    glossary = {"binary tree": "Binary Tree"}
    c = TerminologyConstraint(glossary=glossary, penalty=1.0)
    assert c.score("binary tree is a data structure") == 0.0
    assert c.score("tree structure is a data structure") > 0.0
