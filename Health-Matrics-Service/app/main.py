"""
Main entry point for the Health Metrics Service.

This script initializes the FastAPI application, loads environment variables,
and includes the API router for all the service's endpoints.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from app import routes
from app.scheduler import delete_old_health_reports
from apscheduler.schedulers.background import BackgroundScheduler

load_dotenv()

app = FastAPI(title="Medical Report OCR Service")

scheduler = BackgroundScheduler()

@app.on_event("startup")
def start_scheduler():
    # Schedule the job to run every day at a specific time (e.g., 2:00 AM)
    scheduler.add_job(delete_old_health_reports, 'cron', hour=2, minute=00)
    scheduler.start()
    print("Scheduler started...")

@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()
    print("Scheduler shut down...")

app.include_router(routes.router, prefix="/api")