
from unittest.mock import patch

from app.models.document import DocumentPage, ExtractedDocument
from app.services.document_embedding_service import (
    prepare_document_embeddings
)


def test_prepare_document_embeddings():

    document = ExtractedDocument(
        file_name="sample.txt",
        file_type="txt",
        file_path="sample.txt",
        pages=[
            DocumentPage(
                page_number=1,
                text="Artificial intelligence is useful. " * 100
            )
        ]
    )

    fake_vector = [0.1] * 384

    with patch(
        "app.services.document_embedding_service.generate_embeddings",
        return_value=[fake_vector]
    ):

        with patch(
            "app.services.document_embedding_service.chunk_document"
        ) as mock_chunk:

            from app.services.chunking_service import DocumentChunk

            mock_chunk.return_value = [
                DocumentChunk(
                    chunk_id="chunk-123",
                    document_id="doc-123",
                    file_name="sample.txt",
                    page_number=1,
                    chunk_index=0,
                    text="Artificial intelligence is useful."
                )
            ]

            result = prepare_document_embeddings(
                document,
                "doc-123"
            )

    assert len(result) == 1
    assert result[0].document_id == "doc-123"
    assert result[0].page_number == 1
    assert len(result[0].embedding) == 384
