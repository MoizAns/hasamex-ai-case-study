from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import GEMINI_EMBEDDING_MODEL, GOOGLE_API_KEY, require_api_key


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Return a configured Gemini embeddings client."""
    require_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
