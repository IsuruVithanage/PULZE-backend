"""
Centralized application configuration.

This module uses Pydantic's BaseSettings to load configuration from
environment variables and a .env file, providing a single, type-safe
source of truth for all settings.
"""

from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # --- Core API Settings ---
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Health Diet Recommendation API"
    USER_SERVICE_URL: str
    HEALTH_METRICS_SERVICE_URL: str

    # --- Language Model (Groq) Settings ---
    GROQ_API_KEY: str
    LLM_MODEL_NAME: str = "openai/gpt-oss-20b"
    TEMPERATURE: float = 0.1

    # --- Vector Store (Pinecone) Settings ---
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "health-recommendations"

    # --- Embedding Model Settings ---
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768

    # --- File Storage (AWS S3) Settings ---
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # --- RAG & Document Processing Settings ---
    PDF_DIRECTORY: str = "data"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    CONTEXTUAL_CHUNK_SIMILARITY_THRESHOLD: float = 0.75
    RETRIEVER_K: int = 5  # Number of relevant document chunks to retrieve for context

    # --- Environment & System Settings ---
    TOKENIZERS_PARALLELISM: Optional[str] = "false"  # Suppresses a HuggingFace warning

    # --- Pydantic Model Configuration ---
    class Config:
        """Loads settings from the specified .env file."""
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single, globally accessible settings instance
settings = Settings()