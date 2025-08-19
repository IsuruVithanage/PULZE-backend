from pydantic import BaseModel
from datetime import datetime

class ReportBase(BaseModel):
    total_cholesterol: float
    hdl_cholesterol: float
    triglycerides: float
    ldl_cholesterol: float
    vldl_cholesterol: float
    non_hdl_cholesterol: float
    total_hdl_ratio: float
    triglycerides_hdl_ratio: float

class ReportCreate(ReportBase):
    pass

class ReportResponse(ReportBase):
    id: int
    user_id: int
    updated_at: datetime

    class Config:
        from_attributes = True