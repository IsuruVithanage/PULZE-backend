"""
Defines Pydantic schemas for API data validation and serialization.

These models act as the data contract for the API, ensuring that requests
and responses conform to a defined structure.
"""

from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
from enum import Enum


class ReportType(str, Enum):
    """Enum to enforce a specific set of report types in the API."""
    LIPID = "lipid"
    BLOOD_SUGAR = "blood_sugar"


# --- Schemas for Lipid Report ---
class LipidReportBase(BaseModel):
    """Base schema for lipid report data."""
    total_cholesterol: float
    hdl_cholesterol: float
    triglycerides: float
    ldl_cholesterol: float
    vldl_cholesterol: Optional[float] = None
    non_hdl_cholesterol: Optional[float] = None
    total_hdl_ratio: Optional[float] = None
    triglycerides_hdl_ratio: Optional[float] = None


class LipidReportResponse(LipidReportBase):
    """Schema for responding with lipid report data, including database fields."""
    id: int
    user_id: int
    updated_at: datetime

    class Config:
        from_attributes = True


# --- Schemas for Blood Sugar Report ---
class BloodSugarReportBase(BaseModel):
    """Base schema for blood sugar report data."""
    fasting_blood_sugar: Optional[float] = None
    random_blood_sugar: Optional[float] = None
    hba1c: Optional[float] = None


class BloodSugarReportResponse(BloodSugarReportBase):
    """Schema for responding with blood sugar data, including database fields."""
    id: int
    user_id: int
    updated_at: datetime

    class Config:
        from_attributes = True


# --- Schemas for Utility Endpoints ---
class UpdateMetricRequest(BaseModel):
    """Schema for updating a single metric in a report."""
    report_type: ReportType
    metric_name: str
    metric_value: float


class ChartDataset(BaseModel):
    """Schema for a dataset within a chart response."""
    data: List[Optional[float]]


class ChartResponse(BaseModel):
    """Schema for providing data formatted for charting libraries."""
    labels: List[str]
    datasets: List[ChartDataset]


class CombinedReportResponse(BaseModel):
    """Schema for the combined latest report, including calculated BMI."""
    bmi: Optional[float] = None
    weight: Optional[float] = None
    total_cholesterol: Optional[float] = None
    hdl_cholesterol: Optional[float] = None
    triglycerides: Optional[float] = None
    ldl_cholesterol: Optional[float] = None
    triglycerides_hdl_ratio: Optional[float] = None
    fasting_blood_sugar: Optional[float] = None
    random_blood_sugar: Optional[float] = None
    hba1c: Optional[float] = None
    last_updated: Optional[datetime] = None


# --- Schemas for Historical Summary ---
class TimeSeriesDataPoint(BaseModel):
    """Represents a single data point in a time series (date and value)."""
    date: datetime
    value: Optional[float]


class MetricTimeSeries(BaseModel):
    """Represents a full time series for a single metric."""
    name: str
    unit: str
    series: List[TimeSeriesDataPoint]


class HistoricalDataResponse(BaseModel):
    """Schema for the historical data summary endpoint."""
    metrics: List[MetricTimeSeries]