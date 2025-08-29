import os

from dotenv import load_dotenv
from fastapi import FastAPI
from app import routes

load_dotenv()

app = FastAPI(title="Medical Report OCR Service")

app.include_router(routes.router, prefix="/api")
