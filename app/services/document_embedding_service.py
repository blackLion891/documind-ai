
from dataclasses import dataclass

from app.models.document import ExtractedDocument
from app.services.chunking_service import chunk_document
from app.services.embedding_service import generate_embeddings
from app.utils.exceptions import EmbeddingServiceError
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddedChunk:
    chunk_id: str
    document_id: str
    file_name: str
    page_number: int
    chunk_index: int
    text: str
    embedding: list[float]


def prepare_document_embeddings(
    document: ExtractedDocument,
    document_id: str
) -> list[EmbeddedChunk]:

    try:
        chunks = chunk_document(
            document=document,
            document_id=document_id
        )

        if not chunks:
            raise ValueError("Document contains no chunks.")

        texts = [chunk.text for chunk in chunks]

        vectors = generate_embeddings(texts)

        if len(chunks) != len(vectors):
            raise EmbeddingServiceError(
                "Chunk and embedding counts do not match."
            )

        embedded_chunks = []

        for chunk, vector in zip(chunks, vectors):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    file_name=chunk.file_name,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    embedding=vector
                )
            )

        logger.info(
            "Prepared %s embedded chunks for %s",
            len(embedded_chunks),
            document.file_name
        )

        return embedded_chunks

    except Exception:
        logger.exception("Document embedding preparation failed")
        raise
