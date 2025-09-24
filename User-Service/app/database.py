"""
Handles database connection setup and session management.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

load_dotenv()

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# Create the SQLAlchemy engine, which manages connections to the database
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative class definitions (our ORM models)
Base = declarative_base()


def get_db():
    """
    FastAPI dependency to create and manage database sessions per request.

    This function yields a database session to the API endpoint and ensures
    it is always closed afterward, even if an error occurs.

    Yields:
        Session: A new SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()