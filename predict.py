import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
import mlflow.pytorch

# Load trained model from MLflow
mlflow_model_path = "models:/fake_job_detector/1"  # Load best version automatically

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer from MLflow
model = mlflow.pytorch.load_model(mlflow_model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model.eval()

def predict(job_description):
    inputs = tokenizer(job_description, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logit
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item() * 100

    label = "Fake Job Posting" if predicted_class == 1 else "Real Job Posting"
    return f"{label} ({confidence:.2f}% confidence)"

# Test on a sample real and fake job posting
sample_real = "Looking for an experienced software engineer to join our team. Must have 5+ years experience in Python."
sample_fake = "This job pays $1000 per week for working 2 hours per day. No experience needed, apply now!"

print("Real Job Test:", predict(sample_real))
print("Fake Job Test:", predict(sample_fake))
