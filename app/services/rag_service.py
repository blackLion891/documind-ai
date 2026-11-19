
from app.services.retrieval_service import search_similar_chunks
from app.services.llm_service import generate_response
from app.prompts.system_prompts import RAG_SYSTEM_PROMPT
from app.utils.logger import get_logger

logger = get_logger(__name__)


def ask_document_question(
    question: str,
    top_k: int = 5,
    conversation_history: list | None = None
) -> dict:
    """
    Answer a question using retrieved document context.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1: Retrieve relevant document chunks.
    retrieved_chunks = search_similar_chunks(
        query=question,
        top_k=top_k
    )
    relevance_threshold = 0.30

    retrieved_chunks = [
                chunk
                for chunk in retrieved_chunks
                if chunk["score"] >= relevance_threshold
            ]
    if not retrieved_chunks:
        return {
            "answer": (
                "No relevant information was found "
                "in the uploaded documents."
            ),
            "sources": []
        }

    # Step 2: Build context from retrieved chunks.
    context_parts = []
    sources = []

    for chunk in retrieved_chunks:
        file_name = chunk["file_name"]
        page_number = chunk["page_number"]

        context_parts.append(
            f"""
Source: {file_name}
Page: {page_number}
Content:
{chunk["text"]}
"""
        )

        source = {
            "file_name": file_name,
            "page_number": page_number
        }

        if source not in sources:
            sources.append(source)

    context = "\n\n".join(context_parts)

    # Step 3: Build the prompt for Groq.
    prompt = f"""
Use the following document context to answer the question.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Provide a clear answer with source citations.
"""

    # Step 4: Generate an answer using Groq.
    answer = generate_response(
        prompt=prompt,
        system_prompt=RAG_SYSTEM_PROMPT,
        conversation_history=conversation_history
    )

    logger.info("RAG answer generated successfully")

    # Step 5: Return answer and source information.
    return {
        "answer": answer,
        "sources": sources
    }
