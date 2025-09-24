"""
Handles configuration management for the service.
"""

import os

# Retrieves the database connection URL from environment variables,
# with a default value for local development.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost/medical_ocr_db")
