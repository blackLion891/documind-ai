
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.utils.exceptions import EmbeddingServiceError
from app.utils.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)

    try:
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as error:
        logger.exception("Could not load embedding model")
        raise EmbeddingServiceError(
            "Failed to load the embedding model."
        ) from error


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Convert a batch of text chunks into normalized embedding vectors."""

    if not texts:
        raise ValueError("Text list cannot be empty.")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every text must be a non-empty string.")

    try:
        model = get_embedding_model()

        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        if len(embeddings) != len(texts):
            raise EmbeddingServiceError(
                "The model returned an unexpected number of embeddings."
            )

        vectors = embeddings.tolist()

        logger.info(
            "Generated %s embeddings with %s dimensions each",
            len(vectors),
            len(vectors[0])
        )

        return vectors

    except EmbeddingServiceError:
        raise
    except Exception as error:
        logger.exception("Embedding generation failed")
        raise EmbeddingServiceError(
            "Failed to generate document embeddings."
        ) from error
