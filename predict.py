import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load trained model and tokenizer
model_path = "models/fake_job_detector"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
model.eval()  # Set model to evaluation mode

def predict_job_posting(job_description):
    """Predicts if a job posting is real or fake."""
    inputs = tokenizer(job_description, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move input to GPU if available

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()  # Get the predicted class (0 or 1)

    return "Fake Job Posting" if prediction == 1 else "Real Job Posting"

if __name__ == "__main__":
    print("Enter a job description to classify (or type 'exit' to quit):")
    while True:
        job_desc = input("\nJob Description: ")
        if job_desc.lower() == "exit":
            break
        prediction = predict_job_posting(job_desc)
        print(f"Prediction: {prediction}")
