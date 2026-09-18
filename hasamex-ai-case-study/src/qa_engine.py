from typing import Dict, List

from src.llm_client import LLMResponseError, call_llm_for_json
from src.prompts import ASK_TRANSCRIPTS_SYSTEM_PROMPT, build_ask_transcripts_prompt
from src.quote_validator import find_supporting_segment
from src.retriever import retrieve_segments

INSUFFICIENT_EVIDENCE_MESSAGE = "Insufficient evidence in the transcripts."


def answer_question(vector_store, question: str, k: int = 8) -> Dict:
    """Answer a free-form question grounded only in retrieved transcript evidence."""
    segments = retrieve_segments(vector_store, question, k=k)

    if not segments:
        return {
            "question": question,
            "insufficient_evidence": True,
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
        }

    try:
        raw = call_llm_for_json(
            ASK_TRANSCRIPTS_SYSTEM_PROMPT,
            build_ask_transcripts_prompt(question, segments),
        )
    except LLMResponseError:
        return {
            "question": question,
            "insufficient_evidence": True,
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
        }

    insufficient = bool(raw.get("insufficient_evidence", False))
    answer = str(raw.get("answer", "") or "")
    proposed_quotes = raw.get("quotes", []) or []

    verified_sources: List[Dict] = []
    for item in proposed_quotes:
        quote = str(item.get("quote", "") or "")
        if not quote:
            continue
        match = find_supporting_segment(quote, segments)
        if match:
            verified_sources.append(
                {
                    "quote": match.quote,
                    "expert_name": match.expert_name,
                    "market": match.market,
                    "timestamp": match.timestamp,
                    "speaker": match.speaker,
                    "source_file": match.source_file,
                }
            )
        # Unverifiable quotes are silently dropped rather than shown - the
        # answer text itself is still evaluated on its own merits below.

    if not insufficient and not verified_sources:
        # The model claimed sufficient evidence but produced no quote we can
        # verify - treat this conservatively as insufficient evidence.
        insufficient = True

    return {
        "question": question,
        "insufficient_evidence": insufficient,
        "answer": INSUFFICIENT_EVIDENCE_MESSAGE if insufficient else answer,
        "sources": verified_sources,
    }
