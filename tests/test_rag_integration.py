
from app.models.document import DocumentPage, ExtractedDocument
from app.services.document_embedding_service import (
    prepare_document_embeddings
)
from app.services.vector_store_service import store_embedded_chunks
from app.services.rag_service import ask_document_question
from app.services.vector_db import close_qdrant_client


def test_real_rag_pipeline():
    document = ExtractedDocument(
        file_name="python_guide.txt",
        file_type="txt",
        file_path="python_guide.txt",
        pages=[
            DocumentPage(
                page_number=1,
                text=(
                    "Python is a popular programming language. "
                    "Python is widely used in artificial intelligence. "
                    "Machine learning developers use Python because "
                    "it provides libraries such as NumPy and scikit-learn."
                )
            )
        ]
    )

    try:
        # Step 1: Generate document embeddings.
        embedded_chunks = prepare_document_embeddings(
            document=document,
            document_id="33333333-3333-4333-8333-333333333333"
        )

        # Step 2: Store document chunks in Qdrant.
        stored_count = store_embedded_chunks(embedded_chunks)

        assert stored_count > 0

        # Step 3: Ask a question using the RAG pipeline.
        result = ask_document_question(
            question="Why do machine learning developers use Python?"
        )

        # Step 4: Verify the generated answer.
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

        # Step 5: Verify source information.
        assert len(result["sources"]) > 0
        assert result["sources"][0]["file_name"] == (
            "python_guide.txt"
        )

    finally:
        close_qdrant_client()
