"""
Parses the plain-text expert-call transcripts into structured segments.

The supplied transcripts follow a consistent shape:

    Expert 1 - Dr. Jean Martin
    Role: Head of Urology
    Market: France

    00:00
    Interviewer: Thanks for joining...

    00:18
    Dr. Martin: Adoption is growing...

This module turns that into a list of `TranscriptSegment` objects, each one
tagged with the timestamp, speaker and expert/market metadata it came from.
Timestamps and original wording are preserved exactly so that quotes can
later be verified deterministically against the source text.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Matches "Expert 1 - Dr. Jean Martin" using a hyphen or an en/em dash.
HEADER_NAME_RE = re.compile(r"^Expert\s+\d+\s*[-–—]\s*(?P<name>.+?)\s*$", re.MULTILINE)
HEADER_ROLE_RE = re.compile(r"^Role:\s*(?P<role>.+?)\s*$", re.MULTILINE)
HEADER_MARKET_RE = re.compile(r"^Market:\s*(?P<market>.+?)\s*$", re.MULTILINE)

# Matches a standalone timestamp line like "00:00" or "01:12".
TIMESTAMP_LINE_RE = re.compile(r"^(?P<timestamp>\d{1,2}:\d{2})\s*$", re.MULTILINE)


@dataclass
class Expert:
    name: str
    role: str
    market: str
    source_file: str


@dataclass
class TranscriptSegment:
    expert_name: str
    role: str
    market: str
    source_file: str
    segment_index: int
    timestamp: str
    speaker: str
    text: str


class TranscriptParsingError(Exception):
    """Raised when a transcript file does not match the expected format."""


def _parse_header(raw_text: str, source_file: str) -> Expert:
    name_match = HEADER_NAME_RE.search(raw_text)
    role_match = HEADER_ROLE_RE.search(raw_text)
    market_match = HEADER_MARKET_RE.search(raw_text)

    if not (name_match and role_match and market_match):
        raise TranscriptParsingError(
            f"Could not find Expert/Role/Market header in {source_file}. "
            "Expected lines like 'Expert 1 - Name', 'Role: ...', 'Market: ...'."
        )

    return Expert(
        name=name_match.group("name").strip(),
        role=role_match.group("role").strip(),
        market=market_match.group("market").strip(),
        source_file=source_file,
    )


def _split_dialogue_blocks(raw_text: str) -> List[Tuple[str, str]]:
    """Return a list of (timestamp, block_text) pairs found after the header."""
    matches = list(TIMESTAMP_LINE_RE.finditer(raw_text))
    if not matches:
        raise TranscriptParsingError("No timestamp lines (e.g. '00:00') were found.")

    blocks = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        timestamp = match.group("timestamp")
        block_text = raw_text[start:end].strip()
        if block_text:
            blocks.append((timestamp, block_text))
    return blocks


def _split_speaker_and_text(block_text: str) -> Tuple[str, str]:
    """Split 'Speaker: the rest of the line(s)' into (speaker, text)."""
    # The speaker label is everything before the first colon on the first line.
    first_line, _, remainder = block_text.partition("\n")
    if ":" not in first_line:
        raise TranscriptParsingError(
            f"Expected 'Speaker: text' format, got: {block_text[:60]!r}"
        )
    speaker, _, first_line_text = first_line.partition(":")
    full_text = (first_line_text.strip() + " " + remainder.strip()).strip()
    return speaker.strip(), full_text


def parse_transcript(file_path: Path) -> Tuple[Expert, List[TranscriptSegment]]:
    """Parse a single transcript file into an Expert and its segments."""
    raw_text = Path(file_path).read_text(encoding="utf-8")
    source_file = Path(file_path).name

    expert = _parse_header(raw_text, source_file)
    blocks = _split_dialogue_blocks(raw_text)

    segments: List[TranscriptSegment] = []
    for index, (timestamp, block_text) in enumerate(blocks):
        speaker, text = _split_speaker_and_text(block_text)
        segments.append(
            TranscriptSegment(
                expert_name=expert.name,
                role=expert.role,
                market=expert.market,
                source_file=source_file,
                segment_index=index,
                timestamp=timestamp,
                speaker=speaker,
                text=text,
            )
        )

    if not segments:
        raise TranscriptParsingError(f"No dialogue segments found in {source_file}.")

    return expert, segments


def parse_all_transcripts(
    file_paths: List[Path],
) -> List[Tuple[Expert, List[TranscriptSegment]]]:
    """Parse every transcript file, raising with a clear file name on failure."""
    results = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")
        try:
            results.append(parse_transcript(path))
        except TranscriptParsingError as exc:
            raise TranscriptParsingError(f"{path.name}: {exc}") from exc
    return results


def parse_interview_guide(file_path: Path) -> List[str]:
    """Extract the numbered questions from the interview guide text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Interview guide not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    question_re = re.compile(r"^\s*\d+\.\s*(?P<question>.+?)\s*$", re.MULTILINE)
    questions = [m.group("question").strip() for m in question_re.finditer(raw_text)]

    if not questions:
        raise TranscriptParsingError(
            f"No numbered questions (e.g. '1. ...') found in {path.name}."
        )
    return questions
