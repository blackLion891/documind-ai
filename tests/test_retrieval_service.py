
from unittest.mock import patch, MagicMock

from app.services.retrieval_service import search_similar_chunks


def test_search_similar_chunks():
    fake_point = MagicMock()

    fake_point.id = "11111111-1111-4111-8111-111111111111"
    fake_point.score = 0.95
    fake_point.payload = {
        "document_id": "document-123",
        "file_name": "sample.pdf",
        "page_number": 1,
        "chunk_index": 0,
        "text": "Artificial intelligence is useful."
    }

    fake_response = MagicMock()
    fake_response.points = [fake_point]

    with patch(
        "app.services.retrieval_service.generate_embeddings",
        return_value=[[0.1] * 384]
    ):
        with patch(
            "app.services.retrieval_service.vector_db.create_collection"
        ):
            with patch(
                "app.services.retrieval_service.vector_db.get_qdrant_client"
            ) as mock_client:

                mock_client.return_value.query_points.return_value = (
                    fake_response
                )

                results = search_similar_chunks(
                    query="What is artificial intelligence?",
                    top_k=5
                )

    assert len(results) == 1
    assert results[0]["file_name"] == "sample.pdf"
    assert results[0]["page_number"] == 1
    assert results[0]["score"] == 0.95
    assert results[0]["text"] == (
        "Artificial intelligence is useful."
    )
