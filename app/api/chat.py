
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_service import ask_document_question
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question about the uploaded documents"
    )
    top_k: int = Field(default=5, ge=1, le=20)


@router.post("/ask")
def ask_question(request: ChatRequest):
    try:
        result = ask_document_question(
            question=request.question,
            top_k=request.top_k
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error

    except Exception as error:
        logger.exception("Chat API request failed")

        raise HTTPException(
            status_code=500,
            detail="Failed to generate an answer."
        ) from error
