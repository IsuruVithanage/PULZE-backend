# app/models.py
from sqlalchemy import Column, Integer, Float, DateTime, func, ForeignKey
from .database import Base

class LipidReport(Base):
    __tablename__ = "lipid_reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)

    total_cholesterol = Column(Float, nullable=False)
    hdl_cholesterol = Column(Float, nullable=False)
    triglycerides = Column(Float, nullable=False)
    ldl_cholesterol = Column(Float, nullable=False)
    vldl_cholesterol = Column(Float, nullable=True)
    non_hdl_cholesterol = Column(Float, nullable=True)
    total_hdl_ratio = Column(Float, nullable=True)
    triglycerides_hdl_ratio = Column(Float, nullable=True)

    updated_at = Column(DateTime(timezone=True), server_default=func.now())

class BloodSugarReport(Base):
    __tablename__ = "blood_sugar_reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)

    fasting_blood_sugar = Column(Float, nullable=True)
    random_blood_sugar = Column(Float, nullable=True)
    hba1c = Column(Float, nullable=True)

    updated_at = Column(DateTime(timezone=True), server_default=func.now())