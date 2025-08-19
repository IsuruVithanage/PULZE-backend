import os

from dotenv import load_dotenv
from fastapi import FastAPI
from app import routes

load_dotenv()

app = FastAPI(title="Medical Report OCR Service")

app.include_router(routes.router, prefix="/api")


print("AWS_ACCESS_KEY_ID:", os.getenv("AWS_ACCESS_KEY_ID"))
print("AWS_REGION:", os.getenv("AWS_REGION"))
print("S3_BUCKET_NAME:", os.getenv("S3_BUCKET_NAME"))