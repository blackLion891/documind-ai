
import json
import uuid
from pathlib import Path

from app.services.document_service import extract_document
from app.services.document_embedding_service import prepare_document_embeddings
from app.services.vector_store_service import store_embedded_chunks
from app.utils.exceptions import DocumentProcessingError
from app.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")


def ingest_document(file_path: str) -> dict:
    try:
        # Step 1: Extract document text.
        document = extract_document(file_path)

        # Step 2: Generate a unique document ID.
        document_id = str(uuid.uuid4())

        # Step 3: Split document and generate embeddings.
        embedded_chunks = prepare_document_embeddings(
            document=document,
            document_id=document_id
        )

        # Step 4: Store embeddings and metadata in Qdrant.
        stored_count = store_embedded_chunks(embedded_chunks)

        # Step 5: Save lightweight document metadata.
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        metadata = {
            "document_id": document_id,
            "file_name": document.file_name,
            "file_type": document.file_type,
            "original_path": document.file_path,
            "page_count": len(document.pages),
            "chunk_count": len(embedded_chunks),
            "stored_chunks": stored_count,
            "embedding_dimensions": (
                len(embedded_chunks[0].embedding)
                if embedded_chunks else 0
            )
        }

        metadata_path = PROCESSED_DIR / f"{document_id}.json"

        metadata_path.write_text(
            json.dumps(metadata, indent=4),
            encoding="utf-8"
        )

        logger.info(
            "Document ingestion completed: %s",
            document.file_name
        )

        return metadata

    except DocumentProcessingError:
        raise

    except Exception as error:
        logger.exception("Document ingestion failed")
        raise DocumentProcessingError(
            "Failed to ingest document."
        ) from error
