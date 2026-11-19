from unittest.mock import patch, MagicMock

import pytest

from app.services.llm_service import generate_response
from app.config import settings
from app.utils.exceptions import (
    ConfigurationError,
    LLMServiceError
)


def test_generate_response_success(monkeypatch):

    monkeypatch.setattr(
        settings,
        "GROQ_API_KEY",
        "test-api-key"
    )

    mock_client = MagicMock()

    mock_client.responses.create.return_value.output_text = (
        "RAG combines retrieval with generation."
    )

    with patch(
        "app.services.llm_service.OpenAI",
        return_value=mock_client
    ):

        result = generate_response("Explain RAG")

    assert result == "RAG combines retrieval with generation."


def test_missing_api_key(monkeypatch):

    monkeypatch.setattr(
        settings,
        "GROQ_API_KEY",
        None
    )

    with pytest.raises(ConfigurationError):

        generate_response("Explain RAG")


def test_empty_prompt(monkeypatch):

    monkeypatch.setattr(
        settings,
        "GROQ_API_KEY",
        "test-api-key"
    )

    with pytest.raises(ValueError):

        generate_response("   ")


def test_empty_response(monkeypatch):

    monkeypatch.setattr(
        settings,
        "GROQ_API_KEY",
        "test-api-key"
    )

    mock_client = MagicMock()

    mock_client.responses.create.return_value.output_text = ""

    with patch(
        "app.services.llm_service.OpenAI",
        return_value=mock_client
    ):

        with pytest.raises(LLMServiceError):

            generate_response("Explain RAG")