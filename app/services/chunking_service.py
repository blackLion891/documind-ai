
import uuid

from dataclasses import dataclass

from llama_index.core.node_parser import SentenceSplitter

from app.models.document import ExtractedDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    file_name: str
    page_number: int
    chunk_index: int
    text: str


def chunk_document(
    document: ExtractedDocument,
    document_id: str
) -> list[DocumentChunk]:

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for page in document.pages:

        if not page.text.strip():
            continue

        page_chunks = splitter.split_text(page.text)

        for chunk_text in page_chunks:

            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                file_name=document.file_name,
                page_number=page.page_number,
                chunk_index=len(chunks),
                text=chunk_text
            )

            chunks.append(chunk)

    logger.info(
        "Created %s chunks from document: %s",
        len(chunks),
        document.file_name
    )

    return chunks
