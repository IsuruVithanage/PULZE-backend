from langchain_community.document_loaders import PyPDFLoader
import os
from typing import List
from langchain_core.documents import Document
from fastapi import UploadFile, HTTPException
import shutil


async def load_pdf_from_path(pdf_path: str) -> List[Document]:
    """Load a PDF file from a file path"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    return loader.load()


async def save_uploaded_pdf(upload_file: UploadFile, destination: str) -> str:
    """Save an uploaded PDF file to the specified destination"""
    if not upload_file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Save the uploaded file
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return destination