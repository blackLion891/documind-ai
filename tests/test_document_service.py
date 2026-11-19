
from pathlib import Path
from docx import Document
from pypdf import PdfWriter

from app.services.document_service import extract_text
import pytest

from app.services.document_service import extract_text
from app.utils.exceptions import DocumentProcessingError
from app.services.document_service import MAX_FILE_SIZE_BYTES

from app.services.document_service import extract_document


def test_reject_directory(tmp_path):
    with pytest.raises(DocumentProcessingError):
        extract_text(str(tmp_path))


def test_reject_large_file(tmp_path):
    file_path = tmp_path / "large.txt"

    file_path.write_bytes(
        b"A" * (MAX_FILE_SIZE_BYTES + 1)
    )

    with pytest.raises(DocumentProcessingError):
        extract_text(str(file_path))


def test_case_insensitive_extension(tmp_path):
    file_path = tmp_path / "sample.TXT"

    file_path.write_text(
        "DocuMind AI test",
        encoding="utf-8"
    )

    result = extract_text(str(file_path))

    assert result == "DocuMind AI test"

def test_extract_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "Welcome to DocuMind AI",
        encoding="utf-8"
    )

    result = extract_text(str(file_path))

    assert result == "Welcome to DocuMind AI"


def test_unsupported_file(tmp_path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("name,age", encoding="utf-8")

    with pytest.raises(DocumentProcessingError):
        extract_text(str(file_path))


def test_missing_file():
    with pytest.raises(DocumentProcessingError):
        extract_text("missing.txt")


def test_empty_txt(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(DocumentProcessingError):
        extract_text(str(file_path))


def test_extract_pdf(tmp_path):
    file_path = tmp_path / "sample.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)

    with open(file_path, "wb") as pdf_file:
        writer.write(pdf_file)

    # A blank PDF has no extractable text.
    from app.utils.exceptions import DocumentProcessingError

    with pytest.raises(DocumentProcessingError):
        extract_text(str(file_path))
def test_extract_docx(tmp_path):
    file_path = tmp_path / "sample.docx"

    document = Document()
    document.add_paragraph("Welcome to DocuMind AI")
    document.add_paragraph("This is a document processing test.")
    document.save(file_path)

    result = extract_text(str(file_path))

    assert "Welcome to DocuMind AI" in result
    assert "This is a document processing test." in result

def test_document_metadata(tmp_path):

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "Welcome to DocuMind AI",
        encoding="utf-8"
    )

    result = extract_document(str(file_path))

    assert result.file_name == "sample.txt"
    assert result.file_type == "txt"
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert result.pages[0].text == "Welcome to DocuMind AI"
    assert result.full_text == "Welcome to DocuMind AI"