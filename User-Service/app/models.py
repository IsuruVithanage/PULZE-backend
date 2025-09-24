"""
Defines the SQLAlchemy ORM models for the database.

Each class in this file represents a table in the database and its columns.
"""

from sqlalchemy import Column, Integer, String, Float, JSON
from .database import Base


class User(Base):
    """
    Represents the 'users' table in the database.
    """
    __tablename__ = "users"

    # Core user identification fields
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    # Basic user profile information
    name = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    weight_kg = Column(Float, nullable=True)
    height_cm = Column(Float, nullable=True)

    # Additional health-related information stored as JSON
    health_conditions = Column(JSON, nullable=True)
    lifestyle_habits = Column(JSON, nullable=True)