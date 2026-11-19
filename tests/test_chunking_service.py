
from app.models.document import DocumentPage, ExtractedDocument
from app.services.chunking_service import chunk_document


def test_chunk_document():
    text = (
        "Artificial intelligence is a technology. "
        "Machine learning is a part of artificial intelligence. "
    ) * 100

    document = ExtractedDocument(
        file_name="sample.txt",
        file_type="txt",
        file_path="sample.txt",
        pages=[
            DocumentPage(page_number=1, text=text),
            DocumentPage(page_number=2, text=text)
        ]
    )

    chunks = chunk_document(
        document=document,
        document_id="test-document-123"
    )

    assert len(chunks) > 0
    assert all(chunk.text.strip() for chunk in chunks)
    assert all(
        chunk.document_id == "test-document-123"
        for chunk in chunks
    )
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
