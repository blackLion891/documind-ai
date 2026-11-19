
from app.services import vector_db


def test_create_qdrant_collection(tmp_path, monkeypatch):

    # Close any previously cached client
    vector_db.close_qdrant_client()

    # Use temporary storage for testing
    monkeypatch.setattr(
        vector_db,
        "QDRANT_PATH",
        tmp_path / "qdrant"
    )

    try:
        # Create collection
        result = vector_db.create_collection()

        assert result == "documind_documents"

        client = vector_db.get_qdrant_client()

        # Verify collection exists
        assert client.collection_exists(
            collection_name="documind_documents"
        )

        # Verify vector configuration
        collection = client.get_collection(
            collection_name="documind_documents"
        )

        assert collection.config.params.vectors.size == 384
        assert collection.config.params.vectors.distance.value == "Cosine"

        # Verify calling it again does not create a duplicate
        second_result = vector_db.create_collection()

        assert second_result == "documind_documents"

    finally:
        vector_db.close_qdrant_client()
