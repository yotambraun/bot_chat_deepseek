import boto3
from format_the_dataset import dataset
import os
from dotenv import load_dotenv
load_dotenv()

bucket_name = os.getenv("BUCKET_NAME")
print(f"Bucket Name: {bucket_name}") 
s3 = boto3.client('s3')
file_name = 'formatted_dataset.json'
dataset.to_json(file_name)

def upload_to_s3(file_name, bucket_name):
    if not bucket_name:
        raise ValueError("Bucket name is empty. Make sure BUCKET_NAME is set in the environment variables.")
    s3.upload_file(file_name, bucket_name, f"datasets/{file_name}")
    print(f"Uploaded {file_name} to {bucket_name}/datasets/{file_name}")

upload_to_s3(file_name, bucket_name)
