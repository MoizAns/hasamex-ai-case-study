GROUNDING_RULE = (
    "Answer only from the supplied transcript evidence. Do not use outside "
    "knowledge. Do not invent facts, quotes or timestamps. If the evidence "
    "is insufficient, say so explicitly."
)


def format_segments_for_prompt(segments: list) -> str:
    """Render retrieved segments as a numbered, labeled evidence block."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(
            f"[Segment {i}] expert={seg['expert_name']} market={seg['market']} "
            f"timestamp={seg['timestamp']} speaker={seg['speaker']}\n"
            f"\"{seg['text']}\""
        )
    return "\n\n".join(lines)


INTERVIEW_QUESTION_SYSTEM_PROMPT = f"""You are an analyst answering interview-guide \
questions about a single expert's call transcript, for a hospital robotic-surgery \
market research project.

{GROUNDING_RULE}

Every quote you provide must be copied EXACTLY, character for character, from the \
supplied segments - no paraphrasing inside the quote field. Do not guess a timestamp; \
only use timestamps that appear in the supplied segments.

Respond with ONLY a JSON object (no markdown fences, no extra prose) in this exact \
shape:
{{
  "insufficient_evidence": <true or false>,
  "answer": "<concise answer in your own words, or empty string if insufficient>",
  "quote": "<the single most relevant exact quote supporting the answer, or empty string>",
  "quote_segment_number": <the [Segment N] number the quote came from, or null>
}}
"""


def build_interview_question_prompt(question: str, segments: list) -> str:
    evidence = format_segments_for_prompt(segments)
    return (
        f"Question: {question}\n\n"
        f"Evidence from this expert's transcript only:\n\n{evidence}\n\n"
        "Answer the question using only this evidence."
    )


CROSS_CALL_SYSTEM_PROMPT = f"""You are an analyst comparing three expert-call \
transcripts about robotic surgery adoption in France, Germany and the United Kingdom.

{GROUNDING_RULE}

Be careful with the word "disagreement". Only call something a disagreement if the \
experts' statements genuinely conflict. If experts simply place different emphasis on \
a topic without contradicting each other, label it "Difference in emphasis" instead.

Every quote must be copied EXACTLY from the supplied segments. Only use timestamps \
that appear in the supplied segments.

Respond with ONLY a JSON object (no markdown fences, no extra prose) in this exact \
shape:
{{
  "common_themes": [
    {{
      "theme": "<short theme name>",
      "explanation": "<1-2 sentence explanation>",
      "supporting_experts": ["<expert name>", ...],
      "evidence": [
        {{"expert_name": "...", "market": "...", "quote": "...", "quote_segment_number": <int>}}
      ]
    }}
  ],
  "differences": [
    {{
      "topic": "<short topic name>",
      "type": "Disagreement" or "Difference in emphasis",
      "explanation": "<why this is classified that way>",
      "positions": [
        {{"expert_name": "...", "market": "...", "position": "...", "quote": "...", "quote_segment_number": <int>}}
      ]
    }}
  ]
}}
"""


def build_cross_call_prompt(segments: list) -> str:
    evidence = format_segments_for_prompt(segments)
    return (
        "Evidence from all three experts (France, Germany, United Kingdom):\n\n"
        f"{evidence}\n\n"
        "Identify common themes and differences/disagreements across the three "
        "experts, using only this evidence."
    )


ASK_TRANSCRIPTS_SYSTEM_PROMPT = f"""You are answering a free-form question about a \
robotic-surgery market research project, using only the retrieved transcript \
evidence supplied to you.

{GROUNDING_RULE}

Every quote must be copied EXACTLY from the supplied segments. Only use timestamps \
that appear in the supplied segments.

Respond with ONLY a JSON object (no markdown fences, no extra prose) in this exact \
shape:
{{
  "insufficient_evidence": <true or false>,
  "answer": "<concise synthesized answer in your own words, or empty string if insufficient>",
  "quotes": [
    {{"quote": "<exact quote>", "quote_segment_number": <int>}}
  ]
}}
"""


def build_ask_transcripts_prompt(question: str, segments: list) -> str:
    evidence = format_segments_for_prompt(segments)
    return (
        f"Question: {question}\n\n"
        f"Retrieved evidence:\n\n{evidence}\n\n"
        "Answer the question using only this evidence. Cite 1-3 supporting quotes."
    )
