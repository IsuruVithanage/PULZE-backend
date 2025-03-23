from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Health Diet Recommendation API"

    # Model settings
    LLM_MODEL: str = "llama3.1"
    TEMPERATURE: float = 0

    # Document settings
    PDF_PATH: str = os.getenv("PDF_PATH", "data/diet.pdf")
    PDF_DIRECTORY: str = os.getenv("PDF_DIRECTORY", "data")
    CHUNK_SIZE: int = 250
    CHUNK_OVERLAP: int = 0

    # Vector store settings
    RETRIEVER_K: int = 4

    class Config:
        env_file = ".env"

    def get_pdf_path(self) -> str:
        """Get the absolute path to the PDF file"""
        # First check if the PDF_PATH is an absolute path
        if os.path.isabs(self.PDF_PATH) and os.path.exists(self.PDF_PATH):
            return self.PDF_PATH

        # Check in the current directory
        if os.path.exists(self.PDF_PATH):
            return self.PDF_PATH

        # Check in the PDF_DIRECTORY
        os.makedirs(self.PDF_DIRECTORY, exist_ok=True)
        pdf_in_dir = os.path.join(self.PDF_DIRECTORY, os.path.basename(self.PDF_PATH))
        if os.path.exists(pdf_in_dir):
            return pdf_in_dir

        # Return the default path (it doesn't exist, but we'll handle that in the service)
        return self.PDF_PATH


settings = Settings()