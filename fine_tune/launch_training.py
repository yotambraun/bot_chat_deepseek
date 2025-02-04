from sagemaker.huggingface import HuggingFace
import os
from dotenv import load_dotenv
load_dotenv()

role = os.getenv("ROLE")
bucket_name = os.getenv("BUCKET_NAME")

huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir=".",    
    role=role,         
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    transformers_version="4.28.1",
    pytorch_version="2.0.0", 
    py_version="py310",
    hyperparameters={
    'num_train_epochs': 3,
    'per_device_train_batch_size': 2,
    'gradient_accumulation_steps': 4,
    'learning_rate': 2e-4,
    'max_steps': 120,
    'output_dir': '/opt/ml/model',
    'bucket_name': bucket_name,
    'model_name': os.getenv("MODEL_NAME") 
},
    output_path=f's3://{bucket_name}/outputs'
)

huggingface_estimator.fit()