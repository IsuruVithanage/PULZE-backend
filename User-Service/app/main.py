"""
Main entry point for the User Authentication Service.

This script initializes the FastAPI application, sets up the database,
and includes the necessary API routers.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from .routes import auth
from . import models
from .database import engine

load_dotenv()

# Create all database tables defined in models.py if they don't exist
models.Base.metadata.create_all(bind=engine)

# Initialize the FastAPI application
app = FastAPI(title="User Authentication Service")

# All routes defined in routes/auth.py will be available under the /api prefix
app.include_router(auth.router, prefix="/api")