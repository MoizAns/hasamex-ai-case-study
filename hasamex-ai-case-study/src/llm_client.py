import json
import re
from typing import Any, Dict

from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GEMINI_MODEL, GEMINI_TEMPERATURE, GOOGLE_API_KEY, require_api_key

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMResponseError(Exception):
    """Raised when the model does not return parseable JSON."""


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a configured Gemini chat client."""
    require_api_key()
    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=GEMINI_TEMPERATURE,
    )


def _strip_code_fences(text) -> str:
    """Convert Gemini/LangChain response content to text and remove code fences."""

    if isinstance(text, list):
        parts = []

        for item in text:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))

        text = "".join(parts)

    elif not isinstance(text, str):
        text = str(text)

    return _CODE_FENCE_RE.sub("", text).strip()


def call_llm_for_json(system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    Call Gemini with a system + user prompt and parse the response as JSON.

    Raises LLMResponseError if the response cannot be parsed, so callers can
    fall back to an "insufficient evidence" style response instead of
    silently trusting malformed output.
    """
    llm = get_llm()
    messages = [
        ("system", system_prompt),
        ("human", user_prompt),
    ]
    response = llm.invoke(messages)
    raw_text = response.content if hasattr(response, "content") else str(response)
    cleaned = _strip_code_fences(raw_text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Try to salvage a JSON object embedded in extra prose.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        raise LLMResponseError(
            f"Model did not return valid JSON. Raw response: {raw_text[:500]}"
        ) from exc
