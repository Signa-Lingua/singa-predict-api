import os

from dotenv import load_dotenv
from google.cloud import storage

load_dotenv()

storage_client = storage.Client()

bucket_name = os.getenv("GCLOUD_STORAGE_BUCKET")
bucket_file_path = os.getenv("MODEL_PATH")


def get_model():
    if os.path.exists(bucket_file_path):
        print(f"Model already exists: {bucket_file_path}")
    else:
        os.makedirs(os.path.dirname(bucket_file_path), exist_ok=True)
        bucket = storage_client.bucket(bucket_name)  # get bucket
        blob = bucket.blob(bucket_file_path)  # get file

        print(f"Downloading model to: {bucket_file_path}")
        blob.download_to_filename(bucket_file_path)

    return bucket_file_path
