from typing import Optional

from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Health Diet Recommendation API"

    # --- NEW: Groq LLM settings ---
    GROQ_API_KEY: str
    LLM_MODEL_NAME: str = "openai/gpt-oss-20b" # Or another model available on Groq
    TEMPERATURE: float = 0

    # --- NEW: Pinecone settings ---
    PINECONE_API_KEY: str
    PINECONE_INDEX_NAME: str = "health-recommendations"

    # --- NEW: Embedding model ---
    # The dimension of this model must match the Pinecone index dimension.
    # 'sentence-transformers/all-mpnet-base-v2' has a dimension of 768.
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768

    # --- NEW: Suppress HuggingFace Tokenizers warning ---
    TOKENIZERS_PARALLELISM: Optional[str] = "false"

    # --- NEW: AWS and S3 Settings ---
    # Add these fields to match your .env file
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str
    S3_BUCKET_NAME: str

    # Document settings
    PDF_DIRECTORY: str = "data"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Vector store settings
    RETRIEVER_K: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()