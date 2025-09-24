"""
Defines the SQLAlchemy ORM models for the database tables.

Each class represents a table and its columns, allowing for structured
storage of parsed health report data.
"""

from sqlalchemy import Column, Integer, Float, DateTime, func
from .database import Base


class LipidReport(Base):
    """Represents the 'lipid_reports' table in the database."""
    __tablename__ = "lipid_reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)

    # Core lipid profile metrics
    total_cholesterol = Column(Float, nullable=False)
    hdl_cholesterol = Column(Float, nullable=False)
    triglycerides = Column(Float, nullable=False)
    ldl_cholesterol = Column(Float, nullable=False)

    # Calculated and additional lipid metrics
    vldl_cholesterol = Column(Float, nullable=True)
    non_hdl_cholesterol = Column(Float, nullable=True)
    total_hdl_ratio = Column(Float, nullable=True)
    triglycerides_hdl_ratio = Column(Float, nullable=True)

    # Timestamp for when the record was last updated
    updated_at = Column(DateTime(timezone=True), server_default=func.now())


class BloodSugarReport(Base):
    """Represents the 'blood_sugar_reports' table in the database."""
    __tablename__ = "blood_sugar_reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)

    # Core blood sugar metrics
    fasting_blood_sugar = Column(Float, nullable=True)
    random_blood_sugar = Column(Float, nullable=True)
    hba1c = Column(Float, nullable=True)

    # Timestamp for when the record was last updated
    updated_at = Column(DateTime(timezone=True), server_default=func.now())