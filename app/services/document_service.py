
from pathlib import Path

from pypdf import PdfReader
from docx import Document

from app.models.document import (
    DocumentPage,
    ExtractedDocument
)
from app.utils.exceptions import DocumentProcessingError
from app.utils.logger import get_logger


logger = get_logger(__name__)

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_document(file_path: str) -> ExtractedDocument:

    path = Path(file_path)

    if not path.exists():
        raise DocumentProcessingError(
            f"File not found: {file_path}"
        )

    if not path.is_file():
        raise DocumentProcessingError(
            f"Path is not a file: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentProcessingError(
            f"Unsupported file format: {extension}"
        )

    if path.stat().st_size > MAX_FILE_SIZE_BYTES:
        raise DocumentProcessingError(
            "File exceeds the maximum allowed size of 20 MB."
        )

    try:
        logger.info(
            "Extracting document: %s",
            path.name
        )

        if extension == ".pdf":
            pages = extract_pdf(path)

        elif extension == ".docx":
            pages = extract_docx(path)

        else:
            pages = extract_txt(path)

        if not any(page.text.strip() for page in pages):
            raise DocumentProcessingError(
                f"No readable text found in: {path.name}"
            )

        return ExtractedDocument(
            file_name=path.name,
            file_type=extension.lstrip("."),
            file_path=str(path),
            pages=pages
        )

    except DocumentProcessingError:
        raise

    except Exception as error:
        logger.exception("Document extraction failed")
        raise DocumentProcessingError(
            f"Could not process file: {path.name}"
        ) from error


def extract_text(file_path: str) -> str:
    document = extract_document(file_path)
    return document.full_text


def extract_pdf(path: Path) -> list[DocumentPage]:

    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(
        reader.pages, start=1
    ):
        text = page.extract_text() or ""

        pages.append(
            DocumentPage(
                page_number=page_number,
                text=text.strip()
            )
        )

    return pages


def extract_docx(path: Path) -> list[DocumentPage]:

    document = Document(str(path))

    paragraphs = [
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    table_text = []

    for table in document.tables:
        for row in table.rows:
            cells = [
                cell.text.strip()
                for cell in row.cells
            ]
            table_text.append(" | ".join(cells))

    text = "\n".join(paragraphs + table_text)

    return [
        DocumentPage(
            page_number=1,
            text=text
        )
    ]


def extract_txt(path: Path) -> list[DocumentPage]:

    text = path.read_text(encoding="utf-8-sig")

    return [
        DocumentPage(
            page_number=1,
            text=text.strip()
        )
    ]
