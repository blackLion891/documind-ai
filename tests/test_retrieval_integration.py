
from app.models.document import DocumentPage, ExtractedDocument
from app.services.document_embedding_service import (
    prepare_document_embeddings
)
from app.services.vector_store_service import store_embedded_chunks
from app.services.retrieval_service import search_similar_chunks
from app.services.vector_db import close_qdrant_client


def test_real_semantic_search():
    document = ExtractedDocument(
        file_name="sample.txt",
        file_type="txt",
        file_path="sample.txt",
        pages=[
            DocumentPage(
                page_number=1,
                text=(
                    "Python is a popular programming language. "
                    "It is widely used in artificial intelligence "
                    "and machine learning applications."
                )
            )
        ]
    )

    try:
        embedded_chunks = prepare_document_embeddings(
            document=document,
            document_id="22222222-2222-4222-8222-222222222222"
        )

        stored_count = store_embedded_chunks(embedded_chunks)

        assert stored_count > 0

        results = search_similar_chunks(
            query="Which programming language is used in machine learning?",
            top_k=3
        )

        assert len(results) > 0
        assert any(
            "Python" in result["text"]
            for result in results
        )

    finally:
        close_qdrant_client()
