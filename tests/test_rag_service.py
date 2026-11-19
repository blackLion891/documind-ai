
from unittest.mock import patch

from app.services.rag_service import ask_document_question


def test_ask_document_question():
    fake_chunks = [
        {
            "chunk_id": "chunk-1",
            "score": 0.95,
            "document_id": "document-1",
            "file_name": "sample.pdf",
            "page_number": 2,
            "chunk_index": 0,
            "text": "Python is used for machine learning."
        }
    ]

    with patch(
        "app.services.rag_service.search_similar_chunks",
        return_value=fake_chunks
    ) as mock_search:

        with patch(
            "app.services.rag_service.generate_response",
            return_value="Python is commonly used for machine learning."
        ) as mock_llm:

            result = ask_document_question(
                question="Which language is used for machine learning?"
            )

    # Check the final answer.
    assert result["answer"] == (
        "Python is commonly used for machine learning."
    )

    # Check the source information.
    assert result["sources"] == [
        {
            "file_name": "sample.pdf",
            "page_number": 2
        }
    ]

    # Check that document retrieval was performed.
    mock_search.assert_called_once()

    # Check that the document context reached the LLM.
    llm_prompt = mock_llm.call_args.kwargs["prompt"]

    assert "Python is used for machine learning." in llm_prompt
    assert "sample.pdf" in llm_prompt
    assert "Page: 2" in llm_prompt
