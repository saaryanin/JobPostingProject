import requests
import json
import numpy as np
import torch
from transformers import AutoTokenizer

# Define the MLflow API endpoint
MLFLOW_ENDPOINT = "http://127.0.0.1:5001/invocations"

# Load the tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Example text input
text_inputs = [
    "This is a great opportunity to work as a Data Scientist at a top company.",
    "Make money fast! Work from home, no experience needed!"
]

# Tokenize inputs the same way as in training
tokenized_inputs = tokenizer(
    text_inputs,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"  # `pt` ensures PyTorch tensors
)

# Convert input tensors to lists, ensuring they are integers (LongTensor)
input_ids = tokenized_inputs["input_ids"].to(torch.long).tolist()  # Cast to int
attention_mask = tokenized_inputs["attention_mask"].to(torch.long).tolist()

# Create the properly formatted JSON request payload
data = {
    "dataframe_split": {
        "columns": [f"feature_{i}" for i in range(128)],  # MLflow expects 128 feature columns
        "data": input_ids  # Ensure we send correctly formatted input IDs
    }
}

# Send request to MLflow model
headers = {"Content-Type": "application/json"}
response = requests.post(MLFLOW_ENDPOINT, data=json.dumps(data), headers=headers)

# Print response
print("Response:", response.json())
