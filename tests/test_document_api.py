
import json

from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app
from app.api import documents

client = TestClient(app)


def test_upload_document(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)

    with patch(
        "app.api.documents.ingest_document"
    ) as mock_ingest:

        mock_ingest.return_value = {
            "document_id": "document-123",
            "file_name": "sample.txt",
            "file_type": "txt",
            "page_count": 1,
            "chunk_count": 1,
            "stored_chunks": 1
        }

        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "sample.txt",
                    b"Python is a programming language.",
                    "text/plain"
                )
            }
        )

    assert response.status_code == 200
    assert response.json()["document"]["file_name"] == "sample.txt"
    assert (upload_dir / "sample.txt").exists()
    mock_ingest.assert_called_once()


def test_upload_invalid_file():
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                "sample.exe",
                b"invalid file",
                "application/octet-stream"
            )
        }
    )

    assert response.status_code == 400


def test_list_documents(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    metadata = {
        "document_id": "document-123",
        "file_name": "sample.txt",
        "file_type": "txt",
        "page_count": 1,
        "chunk_count": 1,
        "stored_chunks": 1
    }

    metadata_file = processed_dir / "document-123.json"

    metadata_file.write_text(
        json.dumps(metadata),
        encoding="utf-8"
    )

    monkeypatch.setattr(
        documents,
        "PROCESSED_DIR",
        processed_dir
    )

    response = client.get("/documents/")

    assert response.status_code == 200
    assert response.json()["total_documents"] == 1
    assert response.json()["documents"][0]["file_name"] == "sample.txt"
