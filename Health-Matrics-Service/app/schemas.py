from pydantic import BaseModel

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

    class Config:
        from_attributes = True