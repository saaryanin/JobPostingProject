import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

# Load trained model
model_path = "models/fake_job_detector"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()

# Test on a sample real and fake job posting
sample_real = "Looking for an experienced software engineer to join our team. Must have 5+ years experience in Python."
sample_fake = "This job pays $1000 per week for working 2 hours per day. No experience needed, apply now!"

def predict(job_description):
    """Predicts if a job posting is real or fake with an adjusted decision threshold."""
    inputs = tokenizer(job_description, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # 🔹 Apply temperature scaling to smooth confidence
        temperature = 1.3  # Reduced from 1.5 to 1.3 for better balance
        probabilities = F.softmax(logits / temperature, dim=1)

        # Get confidence scores
        fake_confidence = probabilities[0, 1].item() * 100
        real_confidence = probabilities[0, 0].item() * 100

        # 🔹 Normalize confidence scores
        if fake_confidence > 50:
            fake_confidence = min(100, fake_confidence * 1.1)  # Slight boost for fake jobs
        if real_confidence > 70:  # 🔹 Lowered threshold from 80% to 70%
            real_confidence = max(70, real_confidence * 0.95)  # Reduce overconfidence slightly

        confidence = max(fake_confidence, real_confidence)

        # 🔹 Adjust final classification rules
        if fake_confidence > 40:
            label = "Fake Job Posting"
        elif real_confidence > 70:  # Lowered from 80% to 70%
            label = "Real Job Posting"
        else:
            label = "Uncertain - More Data Needed"

    return f"{label} ({confidence:.2f}% confidence)"

print("Real Job Test:", predict(sample_real))
print("Fake Job Test:", predict(sample_fake))
