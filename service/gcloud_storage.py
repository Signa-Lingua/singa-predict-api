import os

from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

storage_client = storage.Client()

bucket_name = os.getenv("GCLOUD_STORAGE_BUCKET") 
bucket_file_path = os.getenv("MODEL_PATH")

def get_model():
    bucket = storage_client.get_bucket(bucket_name) # get bucket
    blob = bucket.blob(bucket_file_path) # get file
    model_name = blob.name.split("/")[-1]
    model_file = f"models/{model_name}"

    if os.path.exists(model_file):
        print(f"Model already exists: {model_file}")
        return model_file
    else:
        print(f"Downloading model to: {model_file}")
        blob.download_to_filename(model_file)

    return model_file
