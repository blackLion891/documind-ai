
from app.models.document import DocumentPage, ExtractedDocument
from app.services.document_embedding_service import (
    prepare_document_embeddings
)


def test_complete_embedding_pipeline():

    document = ExtractedDocument(
        file_name="ai_guide.txt",
        file_type="txt",
        file_path="ai_guide.txt",
        pages=[
            DocumentPage(
                page_number=1,
                text=(
                    "Artificial intelligence is transforming technology. "
                    "Machine learning helps computers learn from data. "
                ) * 100
            ),
            DocumentPage(
                page_number=2,
                text=(
                    "Deep learning uses neural networks. "
                    "Natural language processing works with human language. "
                ) * 100
            )
        ]
    )

    embedded_chunks = prepare_document_embeddings(
        document=document,
        document_id="integration-test-001"
    )

    assert len(embedded_chunks) > 0

    for chunk in embedded_chunks:
        assert chunk.document_id == "integration-test-001"
        assert chunk.file_name == "ai_guide.txt"
        assert chunk.page_number in [1, 2]
        assert chunk.text.strip()
        assert len(chunk.embedding) == 384

    assert {chunk.page_number for chunk in embedded_chunks} == {1, 2}
