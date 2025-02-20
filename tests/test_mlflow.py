import requests
import json
import torch
import pandas as pd
from transformers import AutoTokenizer

# ✅ MLflow API Endpoint
MLFLOW_ENDPOINT = "http://127.0.0.1:5001/invocations"

# ✅ Load Tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ✅ Example Input Text
text_inputs = [
    "This is a great opportunity to work as a Data Scientist at a top company.",
    "Make money fast! Work from home, no experience needed!"
]

# ✅ Tokenize Inputs
tokenized_inputs = tokenizer(
    text_inputs,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# ✅ Ensure `torch.int64` (LongTensor)
input_ids = tokenized_inputs["input_ids"].to(dtype=torch.int64).tolist()  # ✅ Fixed dtype
attention_mask = tokenized_inputs["attention_mask"].to(dtype=torch.int64).tolist()  # ✅ Fixed dtype

# ✅ Convert to Pandas DataFrame (Required for MLflow)
df_input = pd.DataFrame(
    input_ids,
    columns=[f"feature_{i}" for i in range(len(input_ids[0]))]
)

# ✅ Convert DataFrame to JSON (MLflow expects `dataframe_split` format)
data = {
    "dataframe_split": df_input.to_dict(orient="split")
}

# ✅ Send Request to MLflow Model
headers = {"Content-Type": "application/json"}
response = requests.post(MLFLOW_ENDPOINT, data=json.dumps(data), headers=headers)

# ✅ Print Response
print("Response:", response.json())
