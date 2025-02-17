import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

print(torch.cuda.is_available())  # Should print True
print(torch.cuda.get_device_name(0))  # Should print "NVIDIA RTX 4060"

# Check for GPU availability
if torch.cuda.is_available():
    print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("GPU not found. Using CPU instead.")
    device = torch.device("cpu")

print(f"Using device: {device}")

# Debugging: Check dataset for missing or invalid values
df = pd.read_csv("datasets/fake_job_postings.csv")
print("Missing descriptions:", df["description"].isnull().sum())
print("Valid string descriptions:", df["description"].apply(lambda x: isinstance(x, str)).value_counts())

# Load dataset
dataset = load_dataset("csv", data_files="datasets/fake_job_postings.csv")["train"]

# Remove empty descriptions
dataset = dataset.filter(
    lambda x: x["description"] is not None and isinstance(x["description"], str) and x["description"].strip() != "")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# Custom PyTorch Dataset class
class JobPostingDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["description"]
        text = "No description provided." if not isinstance(text, str) or text.strip() == "" else text

        tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(int(self.dataset[idx]["fraudulent"]), dtype=torch.long)
        }


def train_model():
    train_dataset = JobPostingDataset(dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Increase batch size for speed

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    optimizer = Adam(model.parameters(), lr=3e-5)

    model.train()
    num_epochs = 1  # Reduce epochs for faster training
    for epoch in range(num_epochs):
        total_loss = 0
        predictions, true_labels = [], []
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU if available
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["label"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            logits = outputs.logits.detach().cpu().numpy()
            predictions.extend(np.argmax(logits, axis=1))
            true_labels.extend(batch["label"].cpu().numpy())

        avg_loss = total_loss / len(train_dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save_pretrained("models/fake_job_detector")
    tokenizer.save_pretrained("models/fake_job_detector")
    print("Model training completed and saved.")


if __name__ == "__main__":
    train_model()
