
from app.services.rag_service import ask_document_question
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationManager:

    def __init__(self):
        self.history = []

    def ask(self, question: str, top_k: int = 5) -> dict:
        result = ask_document_question(
            question=question,
            top_k=top_k,
            conversation_history=self.history
        )

        self.history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        })

        logger.info("Conversation history updated")

        return result

    def get_history(self) -> list:
        return self.history

    def clear_history(self) -> None:
        self.history.clear()
        logger.info("Conversation history cleared")
