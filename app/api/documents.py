
from pathlib import Path
import json
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.ingestion_service import ingest_document
from app.utils.logger import get_logger
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.services import vector_db

logger = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR = Path("data/uploads")
PROCESSED_DIR = Path("data/processed")
MAX_FILE_SIZE = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, DOCX, and TXT files are supported."
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename).name
    file_path = UPLOAD_DIR / safe_filename

    try:
        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds the 20 MB limit."
            )

        file_path.write_bytes(content)

        metadata = ingest_document(str(file_path))

        return {
            "message": "Document uploaded and processed successfully.",
            "document": metadata
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Document upload failed")

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(
            status_code=500,
            detail="Failed to upload and process document."
        ) from error

    finally:
        await file.close()

@router.get("/")
def list_documents():
    """
    Return metadata for all successfully processed documents.
    """

    if not PROCESSED_DIR.exists():
        return {
            "total_documents": 0,
            "documents": []
        }

    documents = []

    try:
        for metadata_file in PROCESSED_DIR.glob("*.json"):
            metadata = json.loads(
                metadata_file.read_text(encoding="utf-8")
            )
            documents.append(metadata)

        return {
            "total_documents": len(documents),
            "documents": documents
        }

    except Exception as error:
        logger.exception("Failed to list documents")

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve documents."
        ) from error

@router.delete("/{document_id}")
def delete_document(document_id: str):
    """
    Delete a document, its Qdrant vectors, and its metadata.
    """

    metadata_path = PROCESSED_DIR / f"{document_id}.json"

    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:
        # Step 1: Read document metadata.
        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        # Step 2: Delete associated vectors from Qdrant.
        client = vector_db.get_qdrant_client()

        if client.collection_exists(
            collection_name=vector_db.COLLECTION_NAME
        ):
            client.delete(
                collection_name=vector_db.COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                wait=True
            )

        # Step 3: Delete metadata.
        metadata_path.unlink()

        # Step 4: Delete the uploaded file if it is
        # inside the project's upload directory.
        original_path = metadata.get("original_path")

        if original_path:
            upload_root = UPLOAD_DIR.resolve()
            uploaded_file = Path(original_path).resolve()

            if (
                uploaded_file.is_relative_to(upload_root)
                and uploaded_file.is_file()
            ):
                uploaded_file.unlink()

        logger.info(
            "Document deleted successfully: %s",
            document_id
        )

        return {
            "message": "Document deleted successfully.",
            "document_id": document_id
        }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Document deletion failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to delete document."
        ) from error
