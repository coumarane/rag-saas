class RAGBaseException(Exception):
    """Base exception for all RAG SaaS application errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class DocumentNotFoundError(RAGBaseException):
    """Raised when a requested document does not exist."""

    def __init__(self, message: str = "Document not found") -> None:
        super().__init__(message=message, status_code=404)


class DocumentProcessingError(RAGBaseException):
    """Raised when document parsing or ingestion fails."""

    def __init__(self, message: str = "Document processing failed") -> None:
        super().__init__(message=message, status_code=422)


class UnsupportedFileTypeError(RAGBaseException):
    """Raised when an uploaded file type is not supported."""

    def __init__(self, message: str = "Unsupported file type") -> None:
        super().__init__(message=message, status_code=415)


class FileTooLargeError(RAGBaseException):
    """Raised when an uploaded file exceeds the maximum allowed size."""

    def __init__(self, message: str = "File too large") -> None:
        super().__init__(message=message, status_code=413)


class VectorStoreError(RAGBaseException):
    """Raised when a Qdrant operation fails."""

    def __init__(self, message: str = "Vector store error") -> None:
        super().__init__(message=message, status_code=503)


class StorageError(RAGBaseException):
    """Raised when an S3/MinIO operation fails."""

    def __init__(self, message: str = "Storage error") -> None:
        super().__init__(message=message, status_code=503)


class LLMError(RAGBaseException):
    """Raised when an LLM API call fails."""

    def __init__(self, message: str = "LLM service error") -> None:
        super().__init__(message=message, status_code=503)


class ConversationNotFoundError(RAGBaseException):
    """Raised when a requested conversation does not exist."""

    def __init__(self, message: str = "Conversation not found") -> None:
        super().__init__(message=message, status_code=404)
