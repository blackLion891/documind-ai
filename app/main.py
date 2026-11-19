
from fastapi import FastAPI

from app.api.health import router as health_router
from app.utils.logger import get_logger
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
from app.api.conversations import router as conversations_router

logger = get_logger(__name__)

app = FastAPI(
    title="DocuMind AI",
    description="AI-powered document question-answering system",
    version="1.0.0"
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(conversations_router)

@app.get("/")
def root():
    return {
        "message": "Welcome to DocuMind AI API",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting DocuMind AI API")

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
