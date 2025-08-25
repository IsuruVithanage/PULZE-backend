import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv # <-- Import the function

# --- IMPORTANT ---
# Load environment variables from .env file before anything else
load_dotenv()
# -----------------

from app.api.api import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for getting diet recommendations based on health metrics using RAG with Pinecone and Groq.",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the main API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
async def root():
    return {"message": f"Welcome to the {settings.PROJECT_NAME}"}

# This part is for direct execution, uvicorn will handle it when run from the terminal
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir="app")