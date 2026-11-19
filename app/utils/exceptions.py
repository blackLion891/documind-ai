class DocuMindException(Exception):
    """Base exception for the DocuMind AI application."""

class ConfigurationError(DocuMindException):
    """Raised when application configuration is invalid."""


class DocumentProcessingError(DocuMindException):
    """Raised when document processing fails."""


class RetrievalError(DocuMindException):
    """Raised when document retrieval fails."""

class LLMServiceError(DocuMindException):
    """Language model service error."""

class EmbeddingServiceError(DocuMindException):
    """Embedding generation error."""