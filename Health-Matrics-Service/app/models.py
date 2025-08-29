# app/models.py
from sqlalchemy import Column, Integer, Float, DateTime, func, ForeignKey
from .database import Base

# Best practice to keep User model separate, but for simplicity we can define it here
# Or assume it exists elsewhere. We will link reports to a user_id.

class LipidReport(Base):
    __tablename__ = "lipid_reports"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False) # In a full app: Column(Integer, ForeignKey("users.id"))

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
    user_id = Column(Integer, index=True, nullable=False) # Foreign key to user

    fasting_blood_sugar = Column(Float, nullable=True)
    random_blood_sugar = Column(Float, nullable=True)
    hba1c = Column(Float, nullable=True)

    updated_at = Column(DateTime(timezone=True), server_default=func.now())