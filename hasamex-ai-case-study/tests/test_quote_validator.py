"""Tests for src.quote_validator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quote_validator import find_supporting_segment, normalize_whitespace  # noqa: E402

SEGMENTS = [
    {
        "text": "Adoption is growing, but it is still concentrated in larger academic hospitals.",
        "timestamp": "00:18",
        "speaker": "Dr. Martin",
        "source_file": "Transcript_1_France.txt",
        "expert_name": "Dr. Jean Martin",
        "market": "France",
    },
    {
        "text": "The biggest issue is still capital budget approval.",
        "timestamp": "01:20",
        "speaker": "Dr. Martin",
        "source_file": "Transcript_1_France.txt",
        "expert_name": "Dr. Jean Martin",
        "market": "France",
    },
]


def test_accepts_a_genuine_quote():
    match = find_supporting_segment(
        "The biggest issue is still capital budget approval.", SEGMENTS
    )
    assert match is not None
    assert match.timestamp == "01:20"
    assert match.speaker == "Dr. Martin"


def test_accepts_genuine_quote_with_harmless_whitespace_differences():
    match = find_supporting_segment(
        "The biggest issue is still\ncapital   budget approval.", SEGMENTS
    )
    assert match is not None
    assert match.timestamp == "01:20"


def test_rejects_a_fabricated_quote():
    match = find_supporting_segment(
        "The main obstacle is regulatory approval from the ministry of health.",
        SEGMENTS,
    )
    assert match is None


def test_rejects_empty_quote():
    assert find_supporting_segment("", SEGMENTS) is None


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace("a   b\n\nc") == "a b c"
