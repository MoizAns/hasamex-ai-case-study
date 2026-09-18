"""Tests for src.transcript_parser."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transcript_parser import (  # noqa: E402
    parse_all_transcripts,
    parse_interview_guide,
    parse_transcript,
)
from src.config import INTERVIEW_GUIDE_PATH, TRANSCRIPT_FILES  # noqa: E402


def test_parse_transcript_extracts_header_and_timestamps():
    expert, segments = parse_transcript(TRANSCRIPT_FILES[0])

    assert expert.name == "Dr. Jean Martin"
    assert expert.role == "Head of Urology"
    assert expert.market == "France"

    assert len(segments) > 0
    first = segments[0]
    assert first.timestamp == "00:00"
    assert first.speaker == "Interviewer"
    assert "robotic surgery adoption in France" in first.text


def test_parse_transcript_preserves_original_wording():
    _, segments = parse_transcript(TRANSCRIPT_FILES[0])
    combined_text = " ".join(s.text for s in segments)
    # A known phrase from the raw transcript must survive parsing unchanged.
    assert "capital budget approval" in combined_text


def test_parse_all_transcripts_returns_three_experts():
    results = parse_all_transcripts(TRANSCRIPT_FILES)
    assert len(results) == 3
    markets = {expert.market for expert, _ in results}
    assert markets == {"France", "Germany", "United Kingdom"}


def test_parse_interview_guide_returns_numbered_questions():
    questions = parse_interview_guide(INTERVIEW_GUIDE_PATH)
    assert len(questions) == 6
    assert questions[0].startswith("How would you describe current adoption")
