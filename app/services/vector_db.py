
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QDRANT_PATH = PROJECT_ROOT / "data" / "qdrant"

COLLECTION_NAME = "documind_documents"
VECTOR_SIZE = 384


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Create and reuse a persistent local Qdrant client."""

    try:
        QDRANT_PATH.mkdir(parents=True, exist_ok=True)

        client = QdrantClient(path=str(QDRANT_PATH))

        logger.info(
            "Connected to local Qdrant storage: %s",
            QDRANT_PATH
        )

        return client

    except Exception:
        logger.exception("Failed to initialize Qdrant")
        raise


def create_collection() -> str:
    """Create the document collection if it does not already exist."""

    client = get_qdrant_client()

    try:
        if client.collection_exists(collection_name=COLLECTION_NAME):
            logger.info(
                "Qdrant collection already exists: %s",
                COLLECTION_NAME
            )
            return COLLECTION_NAME

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        logger.info(
            "Qdrant collection created: %s",
            COLLECTION_NAME
        )

        return COLLECTION_NAME

    except Exception:
        logger.exception("Failed to create Qdrant collection")
        raise


def close_qdrant_client() -> None:
    """Close the Qdrant client and clear its cached instance."""

    if get_qdrant_client.cache_info().currsize:
        client = get_qdrant_client()
        client.close()
        get_qdrant_client.cache_clear()
        logger.info("Qdrant client connection closed")
