
from fastapi.testclient import TestClient

from app.main import app
from app.api import conversations

client = TestClient(app)


def test_get_conversation_history():
    conversations.conversation_manager.clear_history()

    conversations.conversation_manager.history.append({
        "question": "What is Python?",
        "answer": "Python is a programming language.",
        "sources": []
    })

    response = client.get("/conversations/history")

    assert response.status_code == 200
    assert response.json()["total_messages"] == 1


def test_clear_conversation_history():
    conversations.conversation_manager.history.append({
        "question": "Test question",
        "answer": "Test answer",
        "sources": []
    })

    response = client.delete("/conversations/history")

    assert response.status_code == 200
    assert conversations.conversation_manager.get_history() == []
