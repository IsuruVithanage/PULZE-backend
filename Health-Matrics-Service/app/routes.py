from datetime import datetime
from typing import List, Union

import httpx
from dateutil.relativedelta import relativedelta
from groq import Groq

import boto3
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import os, tempfile, requests
from urllib.parse import urlparse

from app import schemas, models
from app.database import SessionLocal, engine
from app.schemas import CombinedReportResponse
from app.utils import ocr, parser

from pydantic import BaseModel, HttpUrl

models.Base.metadata.create_all(bind=engine)
load_dotenv()
router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def download_file_from_url(url: str) -> str:
    """Download file from a public S3 URL into a temp file and return path."""
    parsed = urlparse(url)
    if parsed.scheme not in ["http", "https"]:
        raise ValueError("Only http/https URLs are allowed")

    # Stream to temp file
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        suffix = os.path.splitext(parsed.path)[1]  # preserve extension
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                tmp.write(chunk)
        tmp.close()
        return tmp.name

class S3UploadRequest(BaseModel):
    s3_url: HttpUrl
    user_id: int
    report_type: schemas.ReportType


@router.post("/upload", status_code=201)
async def upload_s3_file_v2(request: S3UploadRequest, db: Session = Depends(get_db)):
    file_location = None
    try:
        file_location = download_file_from_url(str(request.s3_url))
        extracted_text = ocr.get_text_from_any_file(file_location)

        if request.report_type == schemas.ReportType.LIPID:
            report_data = parser.parse_lipid_report(extracted_text)
            new_report = models.LipidReport(user_id=request.user_id, **report_data)
        elif request.report_type == schemas.ReportType.BLOOD_SUGAR:
            report_data = parser.parse_blood_sugar_report(extracted_text)
            new_report = models.BloodSugarReport(user_id=request.user_id, **report_data)
        else:
            raise HTTPException(status_code=400, detail="Invalid report type specified.")

        db.add(new_report)
        db.commit()
        db.refresh(new_report)
        return new_report

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Parsing failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        if file_location and os.path.exists(file_location):
            os.remove(file_location)


s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

class PresignedRequest(BaseModel):
    user_id: int
    filename: str
    content_type: str

class PresignedUrlResponse(BaseModel):
    upload_url: str
    file_url: str

def sanitize_filename(filename: str) -> str:
    return "".join(c for c in filename if c.isalnum() or c in ("-", "_", ".")) or "file"

@router.post("/s3/presigned", response_model=PresignedUrlResponse)
async def generate_presigned_url(req: PresignedRequest):

    try:
        safe_filename = sanitize_filename(req.filename)
        key = f"uploads/{req.user_id}/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{safe_filename}"

        content_type = str(req.content_type)

        presigned = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": os.getenv('S3_BUCKET_NAME'), "Key": key, "ContentType": content_type},
            ExpiresIn=3600,
        )

        file_url = f"https://{os.getenv('S3_BUCKET_NAME')}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{key}"
        return PresignedUrlResponse(upload_url=presigned, file_url=file_url)
    except Exception as e:
        print("Error generating presigned URL:", e)
        raise HTTPException(status_code=500, detail=str(e))


class FileItem(BaseModel):
    key: str
    url: str
    last_modified: datetime

@router.get("/s3/files/{user_id}", response_model=List[FileItem])
async def list_user_files(user_id: int):
    try:
        prefix = f"uploads/{user_id}/"
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=prefix
        )

        if "Contents" not in response:
            return []

        files = []
        for obj in response["Contents"]:
            key = obj["Key"]
            url = s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET_NAME, "Key": key},
                ExpiresIn=3600,  # 1 hour
            )

            files.append(
                FileItem(
                    key=key.split("/")[-1],
                    url=url,
                    last_modified=obj["LastModified"],
                )
            )

        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {e}")


@router.get("/users/{user_id}/latest-report", response_model=CombinedReportResponse)
def get_latest_combined_report(user_id: int, db: Session = Depends(get_db)):
    """
    Fetches the latest lipid and blood sugar reports for a user and
    combines them into a single response.
    """
    # 1. Fetch the latest lipid report
    latest_lipid_report = (
        db.query(models.LipidReport)
        .filter(models.LipidReport.user_id == user_id)
        .order_by(models.LipidReport.updated_at.desc())
        .first()
    )

    # 2. Fetch the latest blood sugar report
    latest_blood_sugar_report = (
        db.query(models.BloodSugarReport)
        .filter(models.BloodSugarReport.user_id == user_id)
        .order_by(models.BloodSugarReport.updated_at.desc())
        .first()
    )

    # 3. Check if any reports were found at all
    if not latest_lipid_report and not latest_blood_sugar_report:
        raise HTTPException(status_code=404, detail=f"No reports found for user {user_id}")

    # 4. Build the combined response dictionary
    combined_data = {}
    last_updated_time = None

    if latest_lipid_report:
        combined_data.update({
            "total_cholesterol": latest_lipid_report.total_cholesterol,
            "hdl_cholesterol": latest_lipid_report.hdl_cholesterol,
            "triglycerides": latest_lipid_report.triglycerides,
            "ldl_cholesterol": latest_lipid_report.ldl_cholesterol,
            "triglycerides_hdl_ratio": latest_lipid_report.triglycerides_hdl_ratio,
        })
        last_updated_time = latest_lipid_report.updated_at

    if latest_blood_sugar_report:
        combined_data.update({
            "fasting_blood_sugar": latest_blood_sugar_report.fasting_blood_sugar,
            "random_blood_sugar": latest_blood_sugar_report.random_blood_sugar,
            "hba1c": latest_blood_sugar_report.hba1c,
        })
        # Determine which report is more recent for the 'last_updated' field
        if last_updated_time is None or latest_blood_sugar_report.updated_at > last_updated_time:
            last_updated_time = latest_blood_sugar_report.updated_at

    combined_data["last_updated"] = last_updated_time

    return combined_data

try:
    groq_client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
except Exception as e:
    print(f"Error initializing Groq client: {e}")
    groq_client = None

# app/routes.py

# ... (ensure all necessary imports are present)
from app import schemas, models, database
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException


# ... (your other routes) ...

# --- THIS IS THE UPDATED FUNCTION ---
@router.get("/metric/recommendation")
def get_metric_recommendation(
        metric_name: str,
        metric_value: float,
        gender: str,
        age: int,
        weight_kg: float,
        height_cm: float,
        user_id: int,  # <-- NEW: Need user_id to query the database
        db: Session = Depends(get_db)  # <-- NEW: Add database dependency
):
    """
    Provides a brief, AI-generated insight into a user's health metric,
    comparing it with the previous value if available.
    """
    if not groq_client:
        raise HTTPException(
            status_code=500,
            detail="Groq client is not initialized. Check API key."
        )

    # --- NEW LOGIC: Find the previous metric value ---
    previous_value_text = ""
    previous_report = None

    # Determine which table to query based on the metric name
    lipid_metrics = {"Total Cholesterol", "HDL Cholesterol", "LDL Cholesterol", "Triglycerides"}
    sugar_metrics = {"Fasting Blood Sugar", "Random Blood Sugar", "HbA1c"}

    model_to_query = None
    api_metric_name = None

    # A simple mapping from display name to database field name
    metric_map = {
        "Total Cholesterol": "total_cholesterol", "HDL Cholesterol": "hdl_cholesterol",
        "LDL Cholesterol": "ldl_cholesterol", "Triglycerides": "triglycerides",
        "Fasting Blood Sugar": "fasting_blood_sugar", "Random Blood Sugar": "random_blood_sugar",
        "HbA1c": "hba1c"
    }

    api_metric_name = metric_map.get(metric_name)

    if metric_name in lipid_metrics:
        model_to_query = models.LipidReport
    elif metric_name in sugar_metrics:
        model_to_query = models.BloodSugarReport

    if model_to_query and api_metric_name:
        # Query for the second to last report (the one before the latest)
        # We use offset(1) to skip the most recent record and get the one before it.
        previous_report = (
            db.query(model_to_query)
            .filter(model_to_query.user_id == user_id)
            .order_by(model_to_query.updated_at.desc())
            .offset(1)
            .first()
        )

    if previous_report:
        previous_value = getattr(previous_report, api_metric_name, None)
        if previous_value is not None:
            previous_date = previous_report.updated_at.strftime("%B %d, %Y")
            # Create a sentence to add to the prompt
            previous_value_text = f"For comparison, their previous value on {previous_date} was {previous_value}."

    # --- DYNAMIC PROMPT CONSTRUCTION ---
    prompt = f"""
    A user has provided the following health data:
    - Gender: {gender}
    - Age: {age}
    - Weight: {weight_kg} kg
    - Height: {height_cm} cm
    - Health Metric: '{metric_name}'
    - Metric Value: {metric_value}
    {'- Previous Metric Value: ' + previous_value_text if previous_value_text else ''}

    Based on this data, act as a helpful health assistant and provide a brief, easy-to-understand insight into their '{metric_name}'.

    Your response MUST follow these rules:
    1.  Start by stating the user's current metric value and the category it falls into (e.g., "Overweight", "Normal", "High").
    2.  Briefly explain what this means for their health in simple terms.
    3.  Mention the generally accepted healthy range for this metric.
    4.  **If a previous value is provided, briefly compare the current value to the previous one (e.g., "This is an improvement from your previous value of X..." or "This is higher than your previous reading of Y...").**
    5.  Keep the entire response to a single, concise paragraph (about 3-5 sentences).
    6.  Do NOT give medical advice. Focus on explaining the metric.
    7.  Maintain a supportive and encouraging tone.

    Here is an example for a BMI of 28:
    "Your current BMI is 28, which falls in the Overweight range. This means your body weight is higher than what is considered healthy for your height. Maintaining a BMI between 18.5 and 24.9 is ideal and can reduce the risk of developing health issues."

    Here is an example with a previous value:
    "Your current Total Cholesterol is 195, which is in the healthy Normal range. This is a great improvement from your previous reading of 210 on June 15, 2025. Keeping your cholesterol below 200 is excellent for long-term heart health."

    Now, generate the insight for the user's data provided above.
    """

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-120b",
        )
        recommendation = chat_completion.choices[0].message.content
        return {"recommendation": recommendation.strip()}
    except Exception as e:
        print(f"An error occurred with the Groq API: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendation from AI model.")


@router.patch(
    "/users/{user_id}/report/metric",
    # 2. Use Union to specify multiple possible response models
    response_model=Union[schemas.LipidReportResponse, schemas.BloodSugarReportResponse]
)
def update_health_metric(
        user_id: int,
        request: schemas.UpdateMetricRequest,
        db: Session = Depends(get_db),
):
    """
    Updates a single metric in the latest health report for a given user.
    The response model will match the type of report being updated.
    """

    model_to_query = None
    lipid_metrics = {
        "total_cholesterol", "hdl_cholesterol", "triglycerides", "ldl_cholesterol",
        "vldl_cholesterol", "non_hdl_cholesterol", "total_hdl_ratio", "triglycerides_hdl_ratio"
    }
    sugar_metrics = {"fasting_blood_sugar", "random_blood_sugar", "hba1c"}

    if request.metric_name in lipid_metrics:
        model_to_query = models.LipidReport
    elif request.metric_name in sugar_metrics:
        model_to_query = models.BloodSugarReport
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric name: '{request.metric_name}'. Not found in any report type."
        )

    latest_report = (
        db.query(model_to_query)
        .filter(model_to_query.user_id == user_id)
        .order_by(model_to_query.updated_at.desc())
        .first()
    )

    if not latest_report:
        raise HTTPException(
            status_code=404, detail=f"No report of the required type found for user {user_id} to update."
        )

    setattr(latest_report, request.metric_name, request.metric_value)

    try:
        db.commit()
        db.refresh(latest_report)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update report: {e}")

    return latest_report


@router.get("/users/{user_id}/reports/chart-data", response_model=schemas.ChartResponse)
def get_user_report_chart_data(
        user_id: int,
        report_type: schemas.ReportType,  # <-- NEW: Add parameter to specify report type
        metric_name: str,
        months: int = 6,
        db: Session = Depends(get_db),
):
    """
    Retrieves historical data for a specific metric for a given user,
    formatted for use in chart libraries. The report_type determines which table to query.
    """

    # --- NEW: Logic to select the correct model and allowed metrics ---
    model_to_query = None
    allowed_metrics = set()

    if report_type == schemas.ReportType.LIPID:
        model_to_query = models.LipidReport
        allowed_metrics = {
            "total_cholesterol", "hdl_cholesterol", "triglycerides",
            "ldl_cholesterol", "vldl_cholesterol", "non_hdl_cholesterol",
            "total_hdl_ratio", "triglycerides_hdl_ratio"
        }
    elif report_type == schemas.ReportType.BLOOD_SUGAR:
        model_to_query = models.BloodSugarReport
        allowed_metrics = {
            "fasting_blood_sugar", "random_blood_sugar", "hba1c"
        }

    if metric_name not in allowed_metrics:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric name '{metric_name}' for report type '{report_type}'. Charting not supported."
        )

    start_date = datetime.now() - relativedelta(months=months)

    # --- UPDATED: Query the correct model ---
    historical_reports = (
        db.query(model_to_query)
        .filter(model_to_query.user_id == user_id, model_to_query.updated_at >= start_date)
        .order_by(model_to_query.updated_at.asc())
        .all()
    )

    if not historical_reports:
        return {"labels": [], "datasets": [{"data": []}]}

    labels = []
    data_points = []
    for report in historical_reports:
        labels.append(report.updated_at.strftime("%b %d"))
        value = getattr(report, metric_name, None)
        # Return null for missing data points, which chart libraries handle better than 0
        data_points.append(value)

    chart_response = {
        "labels": labels,
        "datasets": [
            {
                "data": data_points
            }
        ]
    }

    return chart_response


@router.get("/users/{user_id}/latest-report")  # Keep your old endpoint name
def get_latest_combined_report(user_id: int, db: Session = Depends(get_db)):
    # ... (your existing logic to fetch latest_lipid_report and latest_blood_sugar_report)
    latest_lipid_report = db.query(models.LipidReport).filter(...).first()
    latest_blood_sugar_report = db.query(models.BloodSugarReport).filter(...).first()

    # --- NEW: Fetch user profile data and calculate BMI ---
    bmi = None
    weight = None
    try:
        # Make a server-to-server call to the User-Service
        profile_url = "http://localhost:8001/api/auth/users/{user_id}/profile"
        with httpx.Client() as client:
            res = client.get(profile_url)
            res.raise_for_status()  # Raise an exception for 4xx/5xx errors
            profile_data = res.json()

        height_cm = profile_data.get("height_cm")
        weight_kg = profile_data.get("weight_kg")
        weight = weight_kg  # To display on the card

        if height_cm and weight_kg:
            height_m = height_cm / 100
            if height_m > 0:
                bmi = round(weight_kg / (height_m * height_m), 1)

    except httpx.RequestError as e:
        print(f"Could not connect to User-Service: {e}")
        # Decide how to handle this: fail the request or continue without BMI?
        # For now, we'll continue without it.
    except Exception as e:
        print(f"Error processing user profile data: {e}")

    # 4. Build the combined response dictionary
    combined_data = {
        "bmi": bmi,
        "weight": weight
    }

    if latest_lipid_report:
        combined_data.update({
            "total_cholesterol": latest_lipid_report.total_cholesterol,
            "hdl_cholesterol": latest_lipid_report.hdl_cholesterol,
            # ... etc for all lipid fields
        })

    if latest_blood_sugar_report:
        combined_data.update({
            "fasting_blood_sugar": latest_blood_sugar_report.fasting_blood_sugar,
            # ... etc for all sugar fields
        })

    return combined_data

@router.get("/users/{user_id}/historical-data", response_model=schemas.HistoricalDataResponse)
def get_historical_data_for_summary(user_id: int, db: Session = Depends(get_db), months: int = 6):
    """
    Fetches a time-series of historical data for key metrics over the last N months.
    """
    metrics_to_track = {
        "Total Cholesterol": {"key": "total_cholesterol", "unit": "mg/dL", "model": models.LipidReport},
        "LDL Cholesterol": {"key": "ldl_cholesterol", "unit": "mg/dL", "model": models.LipidReport},
        "HDL Cholesterol": {"key": "hdl_cholesterol", "unit": "mg/dL", "model": models.LipidReport},
        "Triglycerides": {"key": "triglycerides", "unit": "mg/dL", "model": models.LipidReport},
        "Fasting Blood Sugar": {"key": "fasting_blood_sugar", "unit": "mg/dL", "model": models.BloodSugarReport},
    }

    time_series_data = []
    start_date = datetime.now() - relativedelta(months=months)

    for name, config in metrics_to_track.items():
        model = config["model"]
        metric_key = config["key"]

        # Query for all reports within the date range
        historical_reports = (
            db.query(model)
            .filter(model.user_id == user_id, model.updated_at >= start_date)
            .order_by(model.updated_at.asc())  # Oldest to newest for charting
            .all()
        )

        # Create the list of data points
        series = [
            schemas.TimeSeriesDataPoint(
                date=report.updated_at,
                value=getattr(report, metric_key, None)
            )
            for report in historical_reports
        ]

        time_series_data.append(
            schemas.MetricTimeSeries(
                name=name,
                unit=config["unit"],
                series=series
            )
        )

    return schemas.HistoricalDataResponse(metrics=time_series_data)