import sqlite3
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import VECTORSTORE_DIR
from src.embeddings import get_embeddings

_INDEX_NAME = "transcript_index"


def segment_row_to_metadata(row: sqlite3.Row) -> Dict:
    """Convert a transcript_segments DB row into the metadata dict used everywhere."""
    return {
        "expert_name": row["expert_name"],
        "role": row["role"],
        "market": row["market"],
        "source_file": row["source_file"],
        "segment_index": row["segment_index"],
        "timestamp": row["timestamp"],
        "speaker": row["speaker"],
        "text": row["text"],
    }


def _segments_to_documents(segments: List[sqlite3.Row]) -> List[Document]:
    documents = []
    for row in segments:
        metadata = segment_row_to_metadata(row)
        documents.append(Document(page_content=row["text"], metadata=metadata))
    return documents


def _index_exists() -> bool:
    return (VECTORSTORE_DIR / f"{_INDEX_NAME}.faiss").exists() and (
        VECTORSTORE_DIR / f"{_INDEX_NAME}.pkl"
    ).exists()


def build_vector_store(segments: List[sqlite3.Row]) -> FAISS:
    """Build a fresh FAISS index from transcript segments and persist it to disk."""
    documents = _segments_to_documents(segments)
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTORSTORE_DIR), index_name=_INDEX_NAME)
    return vector_store


def load_vector_store() -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        index_name=_INDEX_NAME,
        # The index only ever contains data we generated ourselves, so
        # deserializing the pickled docstore is safe here.
        allow_dangerous_deserialization=True,
    )


def build_or_load_vector_store(segments: List[sqlite3.Row]) -> FAISS:
    """Reuse a persisted index when available, otherwise build one and save it."""
    if _index_exists():
        return load_vector_store()
    return build_vector_store(segments)
