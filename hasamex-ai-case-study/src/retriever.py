from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS

from src.config import RETRIEVAL_TOP_K


def _document_to_dict(document) -> Dict:
    result = dict(document.metadata)
    result["text"] = document.page_content
    return result


def retrieve_segments(
    vector_store: FAISS,
    query: str,
    k: int = RETRIEVAL_TOP_K,
    expert_name: Optional[str] = None,
    over_fetch_factor: int = 4,
) -> List[Dict]:
    """
    Retrieve the top-k most relevant transcript segments for `query`.

    If `expert_name` is given, results are restricted to that expert's
    transcript by over-fetching and filtering, then trimmed back to k.
    """
    if expert_name:
        fetch_k = max(k * over_fetch_factor, k)
        candidates = vector_store.similarity_search(query, k=fetch_k)
        filtered = [
            doc for doc in candidates if doc.metadata.get("expert_name") == expert_name
        ]
        if len(filtered) < k:
            # Fall back to scanning the full docstore for this expert if the
            # semantic search did not surface enough matches (the transcripts
            # are short, so this is cheap and keeps recall high).
            all_docs = list(vector_store.docstore._dict.values())
            expert_docs = [
                doc for doc in all_docs if doc.metadata.get("expert_name") == expert_name
            ]
            seen_texts = {doc.page_content for doc in filtered}
            for doc in expert_docs:
                if doc.page_content not in seen_texts:
                    filtered.append(doc)
                    seen_texts.add(doc.page_content)
        results = filtered[:k]
    else:
        results = vector_store.similarity_search(query, k=k)

    return [_document_to_dict(doc) for doc in results]


def retrieve_all_for_expert(vector_store: FAISS, expert_name: str) -> List[Dict]:
    """Return every segment for one expert, in original transcript order."""
    all_docs = list(vector_store.docstore._dict.values())
    expert_docs = [
        doc for doc in all_docs if doc.metadata.get("expert_name") == expert_name
    ]
    expert_docs.sort(key=lambda doc: doc.metadata.get("segment_index", 0))
    return [_document_to_dict(doc) for doc in expert_docs]


def retrieve_all_segments(vector_store: FAISS) -> List[Dict]:
    """Return every segment in the index, across all experts."""
    all_docs = list(vector_store.docstore._dict.values())
    all_docs.sort(
        key=lambda doc: (
            doc.metadata.get("source_file", ""),
            doc.metadata.get("segment_index", 0),
        )
    )
    return [_document_to_dict(doc) for doc in all_docs]
