import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost/medical_ocr_db")
