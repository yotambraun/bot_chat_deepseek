# Bot Chat DeepSeek

[![AWS S3](https://raw.githubusercontent.com/ajitgow1997/media/main/s3.png)](#) [![AWS SageMaker](https://raw.githubusercontent.com/ajitgow1997/media/main/sagemaker.png)](#)

Welcome to **Bot Chat DeepSeek** – an AI-powered chatbot system designed to showcase the end-to-end pipeline of fine-tuning a Large Language Model (LLM) on AWS. This repository demonstrates how to prepare a custom dataset, fine-tune a language model, deploy it on AWS SageMaker, and interact with it via a Flask-based API.

### Model (download)
[Download the fine-tuned model](C:\Users\yotam\code_projects\bot_chat_deepseek\model_\model)

### Why LoRA and Parameter-Efficient Tuning?
Traditional full fine-tuning of Large Language Models can be computationally expensive and memory-intensive, especially as model sizes grow into the billions of parameters. Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA (Low-Rank Adaptation of Large Language Models) address these challenges by freezing most of the model’s parameters and only training a small subset of additional weights. This significantly reduces:

1. Memory Footprint: Less GPU memory usage, making fine-tuning feasible on smaller instances.
2. Compute Costs: Shorter training time and fewer resources.
3. Risk of Overfitting: By focusing on smaller parameter sets, the model more robustly adapts to new tasks.

### 4-bit Quantization for Memory Efficiency

Another essential component is loading models in 4-bit precision. This quantization technique reduces memory usage further, allowing large models to be loaded on more affordable AWS instances (e.g., ml.g4dn.xlarge), while retaining acceptable performance. Combining LoRA with 4-bit quantization is especially powerful for prototyping and iterating quickly.

### Supervised Fine-Tuning (SFT) with SFTTrainer
I use the Supervised Fine-Tuning (SFT) approach from the TRL (Training Reinforcement Learning) library by Hugging Face. Despite “RL” in the name, TRL also provides a convenient SFTTrainer class for straightforward supervised training:

1. Supervised Data: We feed a dataset of (instruction, response) pairs into the model to learn direct mappings from task instructions to coherent outputs.
2. Single-Pass Training: SFT is simpler than reinforcement learning from human feedback (RLHF). It’s enough to get a baseline instruction-following capability.
3. Easily Adaptable: You can replace or extend the dataset with domain-specific instructions and responses, enabling the same pipeline to be used for different tasks.
---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [AWS Services Used](#aws-services-used)
4. [Methodology Overview](#methodology-overview)
   - [1. Dataset Preparation](#1-dataset-preparation)
   - [2. Model Fine-Tuning](#2-model-fine-tuning)
   - [3. Model Deployment](#3-model-deployment)
   - [4. Model Interaction](#4-model-interaction)
5. [Installation & Setup](#installation--setup)
6. [Usage](#usage)
7. [Contributing](#contributing)
8. [License](#license)

---

## Features

- **End-to-End Pipeline**: Complete workflow from dataset download to model deployment.
- **Scalable**: Utilizes AWS services for efficient large-scale training and inference.
- **Customizable**: Easily adapt the code to your own datasets, models, and hyperparameters.
- **User-Friendly Interface**: Interact with the model via a simple Flask API endpoint.

---

## Project Structure

```text
.
├── S3
│   ├── download_dataset.py
│   ├── format_the_dataset.py
│   └── upload_to_s3.py
├── fine_tune
│   ├── train.py
│   └── launch_training.py
├── deploy
│   └── deploy.py
├── test_model
│   └── check_model.py
├── app.py
├── README.md
└── requirements.txt
```

* **S3**: Scripts for dataset management (download, format, and upload).
* **fine_tune**: Scripts to train or fine-tune the model on AWS SageMaker.
* **deploy**: Script to deploy the fine-tuned model to an AWS SageMaker endpoint.
* **test_model**: Basic script to check or test the deployed model locally or via endpoint.
* **app.py**: Flask application providing a simple API to interact with the deployed model.
* **requirements.txt**: Lists Python dependencies for this project.

## AWS Services
* **Amazon S3**: For storing the dataset and model artifacts.
* **Amazon SageMaker**: For training and deploying the fine-tuned model.
* **EC2 Backed Instances**: (ml.g4dn.xlarge, ml.g5.xlarge) for cost-effective GPU compute during training and inference.

## Methodology Overview
Below is a more natural and detailed explanation of the methodology used to fine-tune and deploy the chatbot model on AWS:

### 1. Dataset Preparation

#### 1.1. Download the Dataset
* download_dataset.py uses datasets.load_dataset to fetch the Human-Like DPO Dataset from Hugging Face.
* This dataset contains pairs of instructions and responses, making it ideal for conversational fine-tuning.

```python
from datasets import load_dataset

dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split="train")
```

#### 1.2. Format the Dataset
* format_the_dataset.py converts raw data into a prompt-response pair structure.
* The prompts and chosen responses are combined into a single string, simplifying downstream model training.

```python
human_prompt = """Below is an instruction that describes a task. 
Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}
"""

def format_prompts(examples):
    instructions = examples["prompt"]
    outputs = examples["chosen"]
    return {"text": [human_prompt.format(inst, out) 
                     for inst, out in zip(instructions, outputs)]}
```

#### 1.3. Upload to S3
* upload_to_s3.py saves the formatted dataset locally and uploads it to an Amazon S3 bucket.
* This ensures the dataset is easily accessible for SageMaker training jobs.

```python
import boto3
from format_the_dataset import dataset

dataset.to_json('formatted_dataset.json')
s3.upload_file('formatted_dataset.json', bucket_name, 'datasets/formatted_dataset.json')
```

### 2. Model Fine-Tuning
#### 2.1. Training Script
* train.py uses FastLanguageModel (from [unsloth]) and the SFTTrainer (from [trl]) to fine-tune the model on our dataset.
* LoRA-based parameter-efficient tuning is employed for memory-efficient training.
* Key hyperparameters (batch size, learning rate, etc.) are set, and the script executes the fine-tuning process.

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_args.model_name,
    max_seq_length=2048,
    load_in_4bit=True,
)

peft_config = {
    "r": 64,
    "target_modules": [...],
    "lora_alpha": 16,
    ...
}

model = FastLanguageModel.get_peft_model(model, **peft_config)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args
)

trainer.train()
```

#### 2.2. Launch Training on SageMaker
* launch_training.py leverages sagemaker.huggingface.HuggingFace to submit the training job to AWS.
* Specifies the compute instance, role, and output paths for model artifacts.

```python
from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir=".",
    instance_type="ml.g4dn.xlarge",
    ...
    hyperparameters={
        'num_train_epochs': 3,
        'per_device_train_batch_size': 2,
        ...
    }
)

huggingface_estimator.fit()

```

### 3. Model Deployment
#### 3.1. Deploy on SageMaker
* deploy.py uses the trained model artifacts from S3 and creates a SageMaker model.
* An endpoint is then deployed, specifying instance type and environment variables (e.g., MAX_INPUT_LENGTH, MAX_BATCH_SIZE).

```python
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.model import Model

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
        "MAX_BATCH_SIZE": "4"
    }
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    endpoint_name=endpoint_name
)

```

### 4. Model Interaction
#### 4.1. Flask API
* app.py sets up a simple Flask server that sends user queries to the SageMaker endpoint and returns the generated response.
* JSON payload is used to specify generation parameters (e.g., temperature, max_new_tokens).

```python
from flask import Flask, request, jsonify
import boto3
import json

app = Flask(__name__)
sagemaker_runtime = boto3.client('sagemaker-runtime')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    payload = {
        "inputs": user_input,
        "parameters": {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95
        }
    }

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload)
    )
    result = json.loads(response['Body'].read().decode())
    return jsonify({"response": result.get('generated_text', result)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

```

## Installation & Setup
1. Clone the Repository
```bash
git clone https://github.com/yourusername/bot_chat_deepseek.git
cd bot_chat_deepseek
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Configure Environment Variables
Create a file named .env in the project root with the following variables:
```bash
BUCKET_NAME=your-s3-bucket-name
ROLE=arn:aws:iam::123456789012:role/YourSageMakerRole
ENDPOINT_NAME=your-sagemaker-endpoint
MODEL_NAME=your-huggingface-model-name
```

4. Run the Application (Local Testing)
```bash
python app.py
```

The app will be available at http://localhost:5000/chat.


## Usage
1. Interact with the Chatbot
Send a POST request to http://localhost:5000/chat with a JSON body:
```json
{
    "message": "Hello, how can I use this model?"
}
```
You’ll receive a JSON response containing the model’s generated text.
**Fine-Tune Your Own Model**: 
* Modify download_dataset.py to load your custom dataset.
* Adjust format_the_dataset.py for your prompt/response format.
* Update launch_training.py hyperparameters as needed (epochs, learning rate, etc.).
* Deploy with deploy.py and update the endpoint name in .env.