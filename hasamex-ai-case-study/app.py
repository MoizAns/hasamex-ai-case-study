"""
Streamlit UI for the Hasamex AI Engineer take-home case study.

Pages:
    1. Overview
    2. Interview Guide
    3. Cross-Call Analysis
    4. Ask the Transcripts
    5. Methodology / Architecture

Run with:
    streamlit run app.py
"""

import streamlit as st

from src import database
from src.config import (
    ensure_directories,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    INTERVIEW_GUIDE_PATH,
    TRANSCRIPT_FILES,
)
from src.cross_call_analyzer import run_cross_call_analysis
from src.interview_analyzer import run_interview_guide_for_expert
from src.qa_engine import answer_question
from src.transcript_parser import (
    TranscriptParsingError,
    parse_all_transcripts,
    parse_interview_guide,
)
from src.vector_store import build_or_load_vector_store

st.set_page_config(
    page_title="Robotic Surgery Market Intelligence",
    page_icon="🩺",
    layout="wide",
)



@st.cache_resource(show_spinner="Preparing database and vector index...")
def initialize_app_resources():
    """
    First-run setup: parse transcripts into SQLite if needed, then build or
    load the FAISS vector index. Cached so Streamlit reruns (e.g. widget
    interactions) do not repeat this work or re-embed the transcripts.
    """
    ensure_directories()
    database.init_db()

    if not database.is_database_populated():
        parsed = parse_all_transcripts(TRANSCRIPT_FILES)
        for expert, segments in parsed:
            database.insert_expert_and_transcript(expert, segments)

    all_segments = database.get_all_segments()
    vector_store = build_or_load_vector_store(all_segments)
    return vector_store


def get_interview_questions():
    return parse_interview_guide(INTERVIEW_GUIDE_PATH)



def render_evidence_block(result: dict):
    """Render the AI Summary / Supporting Evidence / Timestamp / Source block."""
    if result.get("insufficient_evidence") or not result.get("quote"):
        st.warning(result.get("answer") or "Insufficient evidence in the transcript.")
        return

    st.markdown(f"**AI Summary:** {result['answer']}")
    st.markdown(f"> \u201c{result['quote']}\u201d")
    cols = st.columns(3)
    cols[0].caption(f"⏱️ Timestamp: `{result['timestamp']}`")
    cols[1].caption(f"🗣️ Speaker: {result['speaker']}")
    cols[2].caption(f"📄 Source: {result['source_file']}")


def render_evidence_item(item: dict):
    if not item.get("verified") or not item.get("quote"):
        st.caption("Quote could not be verified against the transcript.")
        return
    st.markdown(f"> \u201c{item['quote']}\u201d")
    st.caption(
        f"{item.get('expert_name', '')} ({item.get('market', '')}) · "
        f"⏱️ `{item.get('timestamp', '')}` · 📄 {item.get('source_file', '')}"
    )




def page_overview(experts, questions):
    st.title("🩺 Robotic Surgery Market Intelligence")
    st.write(
        "An AI-powered analysis of three expert-call transcripts about hospital "
        "adoption of robotic surgery across France, Germany and the United Kingdom. "
        "Every AI-generated claim on this site is traceable back to an exact quote "
        "and timestamp in the source transcripts."
    )

    cols = st.columns(4)
    cols[0].metric("Transcripts", len(experts))
    cols[1].metric("Experts", len(experts))
    cols[2].metric("Markets", len({e["market"] for e in experts}))
    cols[3].metric("Interview questions", len(questions))

    st.subheader("Experts")
    expert_cols = st.columns(len(experts))
    for col, expert in zip(expert_cols, experts):
        with col:
            st.markdown(f"**{expert['name']}**")
            st.caption(expert["role"])
            st.caption(f"Market: {expert['market']}")
            st.caption(f"Source: {expert['source_file']}")


def page_interview_guide(experts, questions, vector_store):
    st.title("📋 Interview Guide")
    st.write(
        "Select an expert to see AI-generated answers to every interview-guide "
        "question, grounded strictly in that expert's own transcript."
    )

    expert_names = [e["name"] for e in experts]
    selected_name = st.selectbox("Select an expert", expert_names)
    force_refresh = st.checkbox("Force refresh (ignore cached answers)", value=False)

    if st.button("Answer all questions", type="primary"):
        with st.spinner(f"Analyzing {selected_name}'s transcript..."):
            results = run_interview_guide_for_expert(
                vector_store, selected_name, questions, force_refresh=force_refresh
            )
        st.session_state[f"interview_results::{selected_name}"] = results

    results = st.session_state.get(f"interview_results::{selected_name}")
    if not results:
        st.info("Click **Answer all questions** to run the analysis.")
        return

    for result in results:
        with st.expander(result["question"], expanded=False):
            render_evidence_block(result)


def page_cross_call(vector_store):
    st.title("🔀 Cross-Call Analysis")
    st.write(
        "Common themes and genuine differences across all three experts. "
        "\"Difference in emphasis\" is used instead of \"disagreement\" unless "
        "the experts' statements actually conflict."
    )

    force_refresh = st.checkbox("Force refresh (ignore cached analysis)", value=False)

    if st.button("Run cross-call analysis", type="primary"):
        with st.spinner("Comparing all three transcripts..."):
            st.session_state["cross_call_result"] = run_cross_call_analysis(
                vector_store, force_refresh=force_refresh
            )

    result = st.session_state.get("cross_call_result")
    if not result:
        st.info("Click **Run cross-call analysis** to generate the comparison.")
        return

    if result.get("error"):
        st.error(result["error"])
        return

    st.subheader("Common Themes")
    themes = result.get("common_themes", [])
    if not themes:
        st.caption("No common themes were identified.")
    for theme in themes:
        with st.expander(f"🧩 {theme['theme']}", expanded=False):
            st.write(theme["explanation"])
            st.caption("Supported by: " + ", ".join(theme.get("supporting_experts", [])))
            for item in theme.get("evidence", []):
                render_evidence_item(item)

    st.subheader("Differences / Disagreements")
    differences = result.get("differences", [])
    if not differences:
        st.caption("No notable differences were identified.")
    for diff in differences:
        badge = "🔴 Disagreement" if diff["type"] == "Disagreement" else "🟡 Difference in emphasis"
        with st.expander(f"{badge}: {diff['topic']}", expanded=False):
            st.write(diff["explanation"])
            table_rows = []
            for position in diff.get("positions", []):
                table_rows.append(
                    {
                        "Expert": position.get("expert_name", ""),
                        "Market": position.get("market", ""),
                        "Position": position.get("position", ""),
                        "Timestamp": position.get("timestamp") or "—",
                    }
                )
            if table_rows:
                st.table(table_rows)
            for position in diff.get("positions", []):
                render_evidence_item(position)


def page_ask_transcripts(vector_store):
    st.title("💬 Ask the Transcripts")
    st.write(
        "Ask any question across all three transcripts. Answers are generated "
        "only from retrieved transcript evidence - never from general knowledge."
    )

    example_questions = [
        "What are the biggest barriers to robotic surgery adoption?",
        "How important is ROI across the three markets?",
        "What do the experts say about training?",
        "How long does purchasing normally take?",
        "How do the UK and Germany differ?",
    ]
    st.caption("Example questions: " + " · ".join(example_questions))

    if "ask_history" not in st.session_state:
        st.session_state["ask_history"] = []

    for turn in st.session_state["ask_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            if turn["insufficient_evidence"]:
                st.warning(turn["answer"])
            else:
                st.write(turn["answer"])
                if turn["sources"]:
                    with st.expander("Supporting evidence"):
                        for source in turn["sources"]:
                            render_evidence_item({**source, "verified": True})

    question = st.chat_input("Ask a question about the transcripts...")
    if question:
        with st.spinner("Retrieving evidence and generating an answer..."):
            result = answer_question(vector_store, question)
        st.session_state["ask_history"].append(result)
        st.rerun()


def page_methodology():
    st.title("🧠 Methodology / Architecture")
    st.markdown(
        """
### Architecture

Transcript files → parser → SQLite → embeddings → FAISS → retriever →
grounded prompt → Gemini → structured response → quote validation →
Streamlit UI.

### Why RAG?

The transcripts are the only source of truth for this project. Retrieval
lets each answer be built from a small, relevant slice of transcript text
rather than asking the model to answer from memory, which keeps every claim
traceable and reduces hallucination risk.

### Model choice

Google Gemini is used for both chat completion and embeddings, configured
through environment variables (`GEMINI_MODEL`, `GEMINI_EMBEDDING_MODEL`) so
the model can be swapped without code changes. Temperature is kept low
(default `0.0`) to favor grounded, repeatable answers.

### Why SQLite?

The relational data here - experts, transcripts, segments, cached analysis
results - is small and local. SQLite needs no separate server, ships with
Python, and is more than sufficient at this scale.

### Why FAISS?

FAISS gives fast, local semantic similarity search over the transcript
segments without needing a hosted vector database, which fits a small
take-home project.

### Citation / timestamp strategy

Timestamps are never generated by the LLM. Every transcript segment stored
in SQLite (and mirrored into FAISS metadata) carries the timestamp read
directly from the transcript file. When the model proposes a supporting
quote, the app searches for that quote inside the retrieved segments; only
if an exact (whitespace-normalized) match is found is the segment's own
timestamp attached to the answer. Unverified quotes are dropped and the
answer is downgraded to "Insufficient evidence in the transcript."

### Hallucination mitigation

- Retrieval-grounded prompts that only include relevant transcript segments.
- Low temperature and explicit "answer only from the supplied evidence" instructions.
- Structured JSON output instead of free text, so responses can be validated.
- A deterministic quote-validation function that rejects any quote not found verbatim in the transcript.
- Timestamps always sourced from segment metadata, never from the model.
- Explicit "Insufficient evidence in the transcript." fallback whenever evidence is missing.

### Limitations

- Retrieval quality depends on the Gemini embedding model and the small size of the transcript corpus.
- Quote validation is whitespace-normalized but otherwise exact-match, so a
  paraphrased "close enough" quote from the model is rejected rather than
  partially credited - this is a deliberate, conservative trade-off.
- Cross-call theme/difference detection is a single LLM call over all
  segments; for larger corpora this would need to be broken into batches.

### Scaling from 3 transcripts to 30+ (future work, not implemented here)

- Background ingestion pipeline instead of first-run parsing in the Streamlit process.
- A persistent, server-based vector database instead of a local FAISS file.
- PostgreSQL instead of SQLite once concurrent writes/multiple users matter.
- Transcript/project IDs and richer metadata filters for multi-project use.
- Batched embedding generation and caching to control API cost.
- Asynchronous processing for ingestion and analysis jobs.
- Stronger evaluation/observability (logging, quote-validation metrics, retrieval quality checks).
        """
    )



def main():
    if not GOOGLE_API_KEY:
        st.error(
            "GOOGLE_API_KEY is not set. Copy `.env.example` to `.env` and add your "
            "Google AI Studio API key, then restart the app."
        )
        st.stop()

    try:
        vector_store = initialize_app_resources()
    except FileNotFoundError as exc:
        st.error(f"Missing transcript file: {exc}")
        st.stop()
    except TranscriptParsingError as exc:
        st.error(f"Could not parse transcripts: {exc}")
        st.stop()
    except Exception as exc:  # noqa: BLE001 - surfaced deliberately to the user
        st.error(f"Failed to initialize the app: {exc}")
        st.stop()

    experts = [dict(e) for e in database.get_all_experts()]
    questions = get_interview_questions()

    st.sidebar.title("Navigation")
    st.sidebar.caption(f"Model: `{GEMINI_MODEL}`")
    st.sidebar.caption(f"Embeddings: `{GEMINI_EMBEDDING_MODEL}`")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Interview Guide",
            "Cross-Call Analysis",
            "Ask the Transcripts",
            "Methodology / Architecture",
        ],
    )

    if page == "Overview":
        page_overview(experts, questions)
    elif page == "Interview Guide":
        page_interview_guide(experts, questions, vector_store)
    elif page == "Cross-Call Analysis":
        page_cross_call(vector_store)
    elif page == "Ask the Transcripts":
        page_ask_transcripts(vector_store)
    else:
        page_methodology()


if __name__ == "__main__":
    main()
