from fastapi import FastAPI
from app import routes

app = FastAPI(title="Medical Report OCR Service")

app.include_router(routes.router, prefix="/api")
