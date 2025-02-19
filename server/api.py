import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# Initialize FastAPI
app = FastAPI()

# Load model
model_path = "models/fake_job_detector"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

class JobDescription(BaseModel):
    text: str

def predict(job_description: str):
    """Predicts if a job posting is real or fake."""
    inputs = tokenizer(job_description, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)

        fake_confidence = probabilities[0, 1].item() * 100
        real_confidence = probabilities[0, 0].item() * 100

        if fake_confidence > 40:
            label = "Fake Job Posting"
        elif real_confidence > 70:
            label = "Real Job Posting"
        else:
            label = "Uncertain - More Data Needed"

    return {"label": label, "confidence": max(fake_confidence, real_confidence)}

@app.post("/predict")
async def predict_api(job: JobDescription):
    """API endpoint to predict job postings."""
    result = predict(job.text)
    return result

@app.get("/")
async def home():
    return {"message": "Fake Job Detector API is running!"}
