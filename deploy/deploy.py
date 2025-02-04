import boto3
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.model import Model
import os
from dotenv import load_dotenv
import json
import logging
import time

logging.basicConfig(
    level='INFO',
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_endpoint_status(endpoint_name):
    """Check the status of a SageMaker endpoint"""
    try:
        sagemaker_client = boto3.client('sagemaker')
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        return response['EndpointStatus']
    except Exception as e:
        logger.error(f"Error checking endpoint status: {str(e)}")
        return None

def delete_endpoint(endpoint_name):
    """Delete a SageMaker endpoint if it exists"""
    try:
        sagemaker_client = boto3.client('sagemaker')
        try:
            sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            logger.info(f"Deleted endpoint config {endpoint_name}")
        except Exception as e:
            logger.info(f"No endpoint config to delete: {str(e)}")
            
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Endpoint {endpoint_name} deletion initiated")
        
        while True:
            try:
                status = get_endpoint_status(endpoint_name)
                if not status:
                    logger.info(f"Endpoint {endpoint_name} deleted successfully")
                    break
                logger.info(f"Waiting for endpoint deletion... Current status: {status}")
                time.sleep(30)
            except:
                logger.info(f"Endpoint {endpoint_name} deleted successfully")
                break
    except Exception as e:
        if "Could not find endpoint" in str(e):
            logger.info(f"Endpoint {endpoint_name} does not exist")
        else:
            logger.error(f"Error deleting endpoint: {str(e)}")
            raise

def deploy_model():
    """Deploy model to SageMaker"""
    load_dotenv()
    role = os.getenv("ROLE")
    endpoint_name = os.getenv("ENDPOINT_NAME")
    bucket_name = os.getenv("BUCKET_NAME")
    
    if not all([role, endpoint_name, bucket_name]):
        raise ValueError("Missing required environment variables. Please check your .env file.")
    
    logger.info(f"Using role: {role}")
    logger.info(f"Using endpoint name: {endpoint_name}")
    model_data = f"s3://{bucket_name}/outputs/huggingface-pytorch-training-2025-02-04-19-03-04-606/output/model.tar.gz"
    logger.info(f"Using model from: {model_data}")
    
    try:
        image_uri = get_huggingface_llm_image_uri(
            backend="huggingface",
            region="us-east-1",
            version="2.4.0"
        )
        model = Model(
            image_uri=image_uri,
            model_data=model_data,
            role=role,
            env={
                "HF_TASK": "text-generation",
                "MAX_INPUT_LENGTH": "1024",
                "MAX_TOTAL_TOKENS": "2048",
                "MAX_BATCH_SIZE": "4",
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20"
            }
        )
        
        logger.info("Starting model deployment (this may take a few minutes)...")
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.g5.xlarge", 
            endpoint_name=endpoint_name,
            wait=True
        )
        
        logger.info("Deployment completed. Testing endpoint...")
        test_data = {
            "inputs": "Tell me a joke!",
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.7,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95
            }
        }
        
        logger.info("Sending test request...")
        response = predictor.predict(
            json.dumps(test_data),
            initial_args={
                "ContentType": "application/json",
                "Accept": "application/json"
            }
        )
        
        logger.info(f"Response from model: {response}")
        logger.info(f"Endpoint is ready at: {endpoint_name}")
        
        return predictor
    
    except Exception as e:
        logger.error(f"Error during deployment: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        load_dotenv()
        endpoint_name = os.getenv("ENDPOINT_NAME")
        current_status = get_endpoint_status(endpoint_name)
        if current_status:
            logger.info(f"Found existing endpoint {endpoint_name} with status {current_status}")
            logger.info("Deleting existing endpoint...")
            delete_endpoint(endpoint_name)
        
        deploy_model()
        
    except Exception as e:
        logger.error("Deployment failed", exc_info=True)
        raise
