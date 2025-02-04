from flask import Flask, request, jsonify
import boto3
import json
from dotenv import load_dotenv
import os


load_dotenv()
app = Flask(__name__)

sagemaker_runtime = boto3.client('sagemaker-runtime')
endpoint_name = os.getenv('ENDPOINT_NAME')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

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
        model_output = result.get('generated_text', result)
        return jsonify({"response": model_output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
