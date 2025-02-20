import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import mlflow.pytorch
import subprocess
import os

# ✅ Load Best Model Path from File
with open("models/best_model.txt", "r") as f:
    best_model_path = f.read().strip()

# ✅ Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# ✅ Load Model & Tokenizer
model = AutoModelForSequenceClassification.from_pretrained(best_model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(best_model_path)

model.eval()  # Set model to evaluation mode

# ✅ Prediction Function
def predict(job_description):
    inputs = tokenizer(job_description, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # ✅ FIX: Correct attribute
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item() * 100

    label = "🚀 Fake Job Posting" if predicted_class == 1 else "✅ Real Job Posting"
    return f"{label} ({confidence:.2f}% confidence)"

# ✅ Test Predictions on Real & Fake Jobs
sample_real = "Looking for an experienced software engineer to join our team. Must have 5+ years experience in Python."
sample_fake = "This job pays $1000 per week for working 2 hours per day. No experience needed, apply now!"

print("🔹 Real Job Test:", predict(sample_real))
print("🔹 Fake Job Test:", predict(sample_fake))

# ✅ Start MLflow UI (Runs in Parallel)
print("\n🚀 Launching MLflow UI at: http://127.0.0.1:5000\n")
subprocess.Popen(["mlflow", "ui"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

