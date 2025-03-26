from sqlalchemy import Column, Integer, Float, String
from .database import Base

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    total_cholesterol = Column(Float, nullable=False)
    hdl_cholesterol = Column(Float, nullable=False)
    triglycerides = Column(Float, nullable=False)
    ldl_cholesterol = Column(Float, nullable=False)
    vldl_cholesterol = Column(Float, nullable=False)
    non_hdl_cholesterol = Column(Float, nullable=False)
    total_hdl_ratio = Column(Float, nullable=False)
    triglycerides_hdl_ratio = Column(Float, nullable=False)
