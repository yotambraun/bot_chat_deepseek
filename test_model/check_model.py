import boto3
import os
from dotenv import load_dotenv
load_dotenv()
s3 = boto3.client('s3')
bucket_name = os.getenv("BUCKET_NAME")
print(f"\nListing contents of bucket: {bucket_name}")
print("----------------------------------------")

try:
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name):
        if 'Contents' in page:
            for obj in page['Contents']:
                print(f"Found: {obj['Key']} ({obj['Size']/1024/1024:.2f} MB)")
        else:
            print("No contents found in bucket")
            
    model_path = "outputs/huggingface-pytorch-training-2025-02-04-19-03-04-606/output/model.tar.gz"
    try:
        response = s3.head_object(Bucket=bucket_name, Key=model_path)
        print(f"\nModel file exists!")
        print(f"Size: {response['ContentLength']/1024/1024:.2f} MB")
        print(f"Last modified: {response['LastModified']}")
        print(f"\nFull S3 path: s3://{bucket_name}/{model_path}")
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            print(f"\nERROR: Model file not found at: {model_path}")
        else:
            print(f"\nError checking model file: {str(e)}")

except Exception as e:
    print(f"Error: {str(e)}")