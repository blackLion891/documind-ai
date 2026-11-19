
from app.services.llm_service import generate_response


class ConversationService:

    def __init__(self):

        self.history = []

    def ask(self, question: str) -> str:

        answer = generate_response(
            prompt=question,
            conversation_history=self.history
        )

        return answer

    def clear_history(self):

        self.history.clear()
