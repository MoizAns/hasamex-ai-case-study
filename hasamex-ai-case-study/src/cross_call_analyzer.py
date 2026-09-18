import json
from typing import Dict, List

from src import database
from src.llm_client import LLMResponseError, call_llm_for_json
from src.prompts import CROSS_CALL_SYSTEM_PROMPT, build_cross_call_prompt
from src.quote_validator import find_supporting_segment
from src.retriever import retrieve_all_segments

CACHE_KEY = "full_analysis"


def _verify_evidence_item(item: Dict, all_segments: List[Dict]) -> Dict:
    """Re-verify a single {expert_name, market, quote, ...} evidence item."""
    quote = str(item.get("quote", "") or "")
    match = find_supporting_segment(quote, all_segments) if quote else None

    verified_item = dict(item)
    if match:
        verified_item["quote"] = match.quote
        verified_item["timestamp"] = match.timestamp
        verified_item["source_file"] = match.source_file
        verified_item["verified"] = True
    else:
        verified_item["quote"] = None
        verified_item["timestamp"] = None
        verified_item["source_file"] = None
        verified_item["verified"] = False
    return verified_item


def _verify_themes(themes: List[Dict], all_segments: List[Dict]) -> List[Dict]:
    verified_themes = []
    for theme in themes:
        evidence = theme.get("evidence", []) or []
        verified_themes.append(
            {
                "theme": theme.get("theme", ""),
                "explanation": theme.get("explanation", ""),
                "supporting_experts": theme.get("supporting_experts", []) or [],
                "evidence": [_verify_evidence_item(e, all_segments) for e in evidence],
            }
        )
    return verified_themes


def _verify_differences(differences: List[Dict], all_segments: List[Dict]) -> List[Dict]:
    verified_differences = []
    for diff in differences:
        positions = diff.get("positions", []) or []
        diff_type = diff.get("type", "Difference in emphasis")
        if diff_type not in ("Disagreement", "Difference in emphasis"):
            diff_type = "Difference in emphasis"
        verified_differences.append(
            {
                "topic": diff.get("topic", ""),
                "type": diff_type,
                "explanation": diff.get("explanation", ""),
                "positions": [_verify_evidence_item(p, all_segments) for p in positions],
            }
        )
    return verified_differences


def run_cross_call_analysis(vector_store, force_refresh: bool = False) -> Dict:
    """Run (or retrieve cached) common-theme and differences analysis."""
    if not force_refresh:
        cached = database.get_cached_analysis_result("cross_call", CACHE_KEY)
        if cached:
            return json.loads(cached)

    all_segments = retrieve_all_segments(vector_store)

    try:
        raw = call_llm_for_json(
            CROSS_CALL_SYSTEM_PROMPT, build_cross_call_prompt(all_segments)
        )
    except LLMResponseError:
        result = {"common_themes": [], "differences": [], "error": "Model response could not be parsed."}
        database.cache_analysis_result("cross_call", CACHE_KEY, json.dumps(result))
        return result

    result = {
        "common_themes": _verify_themes(raw.get("common_themes", []) or [], all_segments),
        "differences": _verify_differences(raw.get("differences", []) or [], all_segments),
    }
    database.cache_analysis_result("cross_call", CACHE_KEY, json.dumps(result))
    return result
