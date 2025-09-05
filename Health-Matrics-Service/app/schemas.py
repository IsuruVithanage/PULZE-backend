# app/schemas.py
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

# --- Enum for Report Type selection in the API ---
class ReportType(str, Enum):
    LIPID = "lipid"
    BLOOD_SUGAR = "blood_sugar"

# --- Schemas for Lipid Report ---
class LipidReportBase(BaseModel):
    total_cholesterol: float
    hdl_cholesterol: float
    triglycerides: float
    ldl_cholesterol: float
    vldl_cholesterol: Optional[float] = None
    non_hdl_cholesterol: Optional[float] = None
    total_hdl_ratio: Optional[float] = None
    triglycerides_hdl_ratio: Optional[float] = None

class LipidReportCreate(LipidReportBase):
    pass

class LipidReportResponse(LipidReportBase):
    id: int
    user_id: int
    updated_at: datetime

    class Config:
        from_attributes = True

# --- Schemas for Blood Sugar Report ---
class BloodSugarReportBase(BaseModel):
    fasting_blood_sugar: Optional[float] = None
    random_blood_sugar: Optional[float] = None
    hba1c: Optional[float] = None

class BloodSugarReportCreate(BloodSugarReportBase):
    pass

class BloodSugarReportResponse(BloodSugarReportBase):
    id: int
    user_id: int
    updated_at: datetime

    class Config:
        from_attributes = True

# --- Schemas for Charting and Metric Updates (can be kept generic) ---
class UpdateMetricRequest(BaseModel):
    report_type: ReportType  # <-- Add this field
    metric_name: str
    metric_value: float

class ChartDataset(BaseModel):
    data: List[Optional[float]]

class ChartResponse(BaseModel):
    labels: List[str]
    datasets: List[ChartDataset]


class CombinedReportResponse(BaseModel):
    total_cholesterol: Optional[float] = None
    hdl_cholesterol: Optional[float] = None
    triglycerides: Optional[float] = None
    ldl_cholesterol: Optional[float] = None
    triglycerides_hdl_ratio: Optional[float] = None
    fasting_blood_sugar: Optional[float] = None
    random_blood_sugar: Optional[float] = None
    hba1c: Optional[float] = None
    last_updated: Optional[datetime] = None



class TimeSeriesDataPoint(BaseModel):
    date: datetime
    value: Optional[float]

class MetricTimeSeries(BaseModel):
    name: str
    unit: str
    series: List[TimeSeriesDataPoint]

class HistoricalDataResponse(BaseModel):
    metrics: List[MetricTimeSeries]