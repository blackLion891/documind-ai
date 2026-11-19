
from app.services import vector_db
from app.services.vector_store_service import store_embedded_chunks
from app.services.document_embedding_service import EmbeddedChunk


def test_store_embedded_chunks(tmp_path, monkeypatch):
    vector_db.close_qdrant_client()

    monkeypatch.setattr(
        vector_db,
        "QDRANT_PATH",
        tmp_path / "qdrant"
    )

    chunks = [
        EmbeddedChunk(
            chunk_id="11111111-1111-4111-8111-111111111111",
            document_id="document-123",
            file_name="sample.pdf",
            page_number=1,
            chunk_index=0,
            text="This is a sample document chunk.",
            embedding=[0.1] * 384
        )
    ]

    try:
        stored_count = store_embedded_chunks(chunks)

        assert stored_count == 1

        client = vector_db.get_qdrant_client()

        stored_points = client.retrieve(
            collection_name=vector_db.COLLECTION_NAME,
            ids=[chunks[0].chunk_id],
            with_payload=True,
            with_vectors=True
        )

        assert len(stored_points) == 1
        assert stored_points[0].payload["file_name"] == "sample.pdf"
        assert stored_points[0].payload["page_number"] == 1
        assert stored_points[0].payload["text"] == (
            "This is a sample document chunk."
        )
        assert len(stored_points[0].vector) == 384

    finally:
        vector_db.close_qdrant_client()
