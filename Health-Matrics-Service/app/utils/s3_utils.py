import boto3
import os
import tempfile
from urllib.parse import urlparse


def download_file_from_s3(s3_url: str) -> str:
    """Download file from S3 URL to a temp file and return its path."""
    parsed_url = urlparse(s3_url)
    bucket = parsed_url.netloc
    key = parsed_url.path.lstrip("/")

    s3 = boto3.client("s3")

    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    s3.download_file(bucket, key, temp_file.name)

    return temp_file.name
