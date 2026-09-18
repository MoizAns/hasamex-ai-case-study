import re
from dataclasses import dataclass
from typing import Dict, List, Optional

WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace so line-wrap differences don't break matching."""
    return WHITESPACE_RE.sub(" ", text).strip()


@dataclass
class QuoteMatch:
    quote: str
    timestamp: str
    speaker: str
    source_file: str
    expert_name: str
    market: str
    segment_text: str


def find_supporting_segment(
    quote: str, segments: List[Dict]
) -> Optional[QuoteMatch]:
    """
    Check whether `quote` occurs verbatim (whitespace-normalized) inside any
    of the given segments. `segments` is a list of dicts with at least the
    keys: text, timestamp, speaker, source_file, expert_name, market.

    Returns the first matching segment's metadata, or None if unsupported.
    """
    if not quote or not quote.strip():
        return None

    normalized_quote = normalize_whitespace(quote)

    for segment in segments:
        normalized_segment_text = normalize_whitespace(segment["text"])
        if normalized_quote in normalized_segment_text:
            return QuoteMatch(
                quote=quote,
                timestamp=segment["timestamp"],
                speaker=segment["speaker"],
                source_file=segment["source_file"],
                expert_name=segment["expert_name"],
                market=segment["market"],
                segment_text=segment["text"],
            )
    return None


def validate_quotes(
    quotes: List[str], segments: List[Dict]
) -> List[Optional[QuoteMatch]]:
    """Validate a batch of proposed quotes against the same candidate segments."""
    return [find_supporting_segment(quote, segments) for quote in quotes]
