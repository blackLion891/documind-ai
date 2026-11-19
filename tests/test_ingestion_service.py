
from unittest.mock import patch

from app.models.document import DocumentPage, ExtractedDocument
from app.services.document_embedding_service import EmbeddedChunk
from app.services.ingestion_service import ingest_document
from app.services.vector_db import close_qdrant_client


def test_ingest_txt_document(tmp_path, monkeypatch):
    try:
        source_file = tmp_path / "sample.txt"
        source_file.write_text(
            "Artificial intelligence is useful.",
            encoding="utf-8"
        )

        processed_dir = tmp_path / "processed"

        monkeypatch.setattr(
            "app.services.ingestion_service.PROCESSED_DIR",
            processed_dir
        )

        fake_document = ExtractedDocument(
            file_name="sample.txt",
            file_type="txt",
            file_path=str(source_file),
            pages=[
                DocumentPage(
                    page_number=1,
                    text="Artificial intelligence is useful."
                )
            ]
        )

        fake_chunk = EmbeddedChunk(
            chunk_id="11111111-1111-4111-8111-111111111111",
            document_id="test-document",
            file_name="sample.txt",
            page_number=1,
            chunk_index=0,
            text="Artificial intelligence is useful.",
            embedding=[0.1] * 384
        )

        with patch(
            "app.services.ingestion_service.extract_document",
            return_value=fake_document
        ):
            with patch(
                "app.services.ingestion_service.prepare_document_embeddings",
                return_value=[fake_chunk]
            ):
                result = ingest_document(str(source_file))

        assert result["file_name"] == "sample.txt"
        assert result["chunk_count"] == 1
        assert result["stored_chunks"] == 1
        assert result["embedding_dimensions"] == 384
        assert result["page_count"] == 1

        metadata_files = list(processed_dir.glob("*.json"))
        assert len(metadata_files) == 1

    finally:
        close_qdrant_client()
