"""Main entry point for the Health Diet Recommendation API."""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.api import api_router
from app.core.config import settings

load_dotenv()
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for diet recommendations using RAG with Pinecone and Groq.",
    version="2.0.0"
)

# --- Middleware ---
# Configure CORS to allow frontend web apps to access this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint for basic health check."""
    return {"message": f"Welcome to the {settings.PROJECT_NAME}"}
