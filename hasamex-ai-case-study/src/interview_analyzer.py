

import json
from typing import Dict, List

from src import database
from src.llm_client import LLMResponseError, call_llm_for_json
from src.prompts import INTERVIEW_QUESTION_SYSTEM_PROMPT, build_interview_question_prompt
from src.quote_validator import find_supporting_segment
from src.retriever import retrieve_all_for_expert, retrieve_segments

INSUFFICIENT_EVIDENCE_MESSAGE = "Insufficient evidence in the transcript."


def _build_result(
    question: str,
    insufficient_evidence: bool,
    answer: str,
    quote_match,
) -> Dict:
    if insufficient_evidence or quote_match is None:
        return {
            "question": question,
            "insufficient_evidence": True,
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE if insufficient_evidence else answer,
            "quote": None,
            "timestamp": None,
            "speaker": None,
            "market": None,
            "source_file": None,
        }
    return {
        "question": question,
        "insufficient_evidence": False,
        "answer": answer,
        "quote": quote_match.quote,
        "timestamp": quote_match.timestamp,
        "speaker": quote_match.speaker,
        "market": quote_match.market,
        "source_file": quote_match.source_file,
    }


def answer_interview_question(
    vector_store, expert_name: str, question: str, force_refresh: bool = False
) -> Dict:
    """Answer one interview-guide question for one expert, with caching."""
    if not force_refresh:
        cached = database.get_cached_analysis_result(
            "interview_guide", question, expert_name=expert_name
        )
        if cached:
            return json.loads(cached)

    # Prefer semantic retrieval, but the transcripts are short (~14 segments
    # each) so falling back to the full transcript keeps recall high.
    candidate_segments = retrieve_segments(
        vector_store, question, k=6, expert_name=expert_name
    )
    if len(candidate_segments) < 4:
        candidate_segments = retrieve_all_for_expert(vector_store, expert_name)

    try:
        raw = call_llm_for_json(
            INTERVIEW_QUESTION_SYSTEM_PROMPT,
            build_interview_question_prompt(question, candidate_segments),
        )
    except LLMResponseError:
        result = _build_result(question, True, "", None)
        database.cache_analysis_result(
            "interview_guide", question, json.dumps(result), expert_name=expert_name
        )
        return result

    insufficient = bool(raw.get("insufficient_evidence", False))
    answer = str(raw.get("answer", "") or "")
    proposed_quote = str(raw.get("quote", "") or "")

    quote_match = None
    if not insufficient and proposed_quote:
        quote_match = find_supporting_segment(proposed_quote, candidate_segments)
        if quote_match is None:
            # The model's quote does not verifiably occur in the transcript.
            # Per the hallucination-safety rules, we do not display it.
            insufficient = True

    result = _build_result(question, insufficient, answer, quote_match)
    database.cache_analysis_result(
        "interview_guide", question, json.dumps(result), expert_name=expert_name
    )
    return result


def run_interview_guide_for_expert(
    vector_store, expert_name: str, questions: List[str], force_refresh: bool = False
) -> List[Dict]:
    """Answer every interview-guide question for one expert."""
    return [
        answer_interview_question(vector_store, expert_name, question, force_refresh)
        for question in questions
    ]
