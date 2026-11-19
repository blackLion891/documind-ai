
from qdrant_client.models import PointStruct

from app.services import vector_db
from app.services.document_embedding_service import EmbeddedChunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


def store_embedded_chunks(chunks: list[EmbeddedChunk]) -> int:
    """
    Store document chunks and their embeddings in Qdrant.

    Returns the number of successfully submitted chunks.
    """

    if not chunks:
        logger.warning("No embedded chunks provided for storage")
        return 0

    try:
        # Make sure the Qdrant collection exists.
        vector_db.create_collection()

        points = []

        for chunk in chunks:
            if len(chunk.embedding) != vector_db.VECTOR_SIZE:
                raise ValueError(
                    f"Invalid embedding size for chunk {chunk.chunk_id}. "
                    f"Expected {vector_db.VECTOR_SIZE}, "
                    f"received {len(chunk.embedding)}."
                )

            point = PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding,
                payload={
                    "document_id": chunk.document_id,
                    "file_name": chunk.file_name,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text
                }
            )

            points.append(point)

        client = vector_db.get_qdrant_client()

        client.upsert(
            collection_name=vector_db.COLLECTION_NAME,
            points=points,
            wait=True
        )

        logger.info(
            "Stored %s document chunks in Qdrant",
            len(points)
        )

        return len(points)

    except Exception:
        logger.exception("Failed to store document embeddings in Qdrant")
        raise
