
import json

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.api import documents
from app.services import vector_db

client = TestClient(app)


def test_delete_document(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    processed_dir = tmp_path / "processed"

    upload_dir.mkdir()
    processed_dir.mkdir()

    document_id = "44444444-4444-4444-8444-444444444444"

    uploaded_file = upload_dir / "sample.txt"
    uploaded_file.write_text("Sample document", encoding="utf-8")

    metadata = {
        "document_id": document_id,
        "file_name": "sample.txt",
        "original_path": str(uploaded_file)
    }

    metadata_file = processed_dir / f"{document_id}.json"
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(documents, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(documents, "PROCESSED_DIR", processed_dir)

    fake_client = MagicMock()
    fake_client.collection_exists.return_value = True

    with patch.object(
        vector_db,
        "get_qdrant_client",
        return_value=fake_client
    ):
        response = client.delete(f"/documents/{document_id}")

    assert response.status_code == 200
    assert not metadata_file.exists()
    assert not uploaded_file.exists()
    fake_client.delete.assert_called_once()


def test_delete_nonexistent_document(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    monkeypatch.setattr(documents, "PROCESSED_DIR", processed_dir)

    response = client.delete(
        "/documents/55555555-5555-4555-8555-555555555555"
    )

    assert response.status_code == 404
