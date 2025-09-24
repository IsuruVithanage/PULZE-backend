"""
Handles database connection setup and session management using SQLAlchemy.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import DATABASE_URL

# Create the main SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a session factory that will be used to create new DB sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class that our ORM models will inherit from
Base = declarative_base()

def get_db():
    """FastAPI dependency to create and manage database sessions per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()