
from fastapi.testclient import TestClient
from unittest.mock import patch

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "application": "DocuMind AI"
    }


def test_chat_endpoint():
    fake_result = {
        "answer": "Python is used for machine learning.",
        "sources": [
            {
                "file_name": "sample.txt",
                "page_number": 1
            }
        ]
    }

    with patch(
        "app.api.chat.ask_document_question",
        return_value=fake_result
    ):
        response = client.post(
            "/chat/ask",
            json={
                "question": "What is Python used for?",
                "top_k": 5
            }
        )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == (
        "Python is used for machine learning."
    )
    assert data["sources"][0]["file_name"] == "sample.txt"


def test_chat_empty_question():
    response = client.post(
        "/chat/ask",
        json={
            "question": "",
            "top_k": 5
        }
    )

    assert response.status_code == 422
