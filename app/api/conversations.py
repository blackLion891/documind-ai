
from fastapi import APIRouter

from app.services.conversation_manager import ConversationManager

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)

conversation_manager = ConversationManager()


@router.get("/history")
def get_conversation_history():
    return {
        "total_messages": len(conversation_manager.get_history()),
        "history": conversation_manager.get_history()
    }


@router.delete("/history")
def clear_conversation_history():
    conversation_manager.clear_history()

    return {
        "message": "Conversation history cleared successfully."
    }
