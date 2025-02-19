import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.pytorch
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training
import time

# Initialize MLflow experiment
mlflow.set_experiment("Fake Job Detector Experiment")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset (limit to 5000 samples for quick testing)
df = pd.read_csv("datasets/fake_job_postings.csv")
df = df.sample(n=5000, random_state=42).reset_index(drop=True)  # Use a subset to speed up training
print(f"Dataset Size: {len(df)} samples")

# Load tokenizer
MODEL_NAME = "bert-base-uncased"  # You can change this to test different models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Custom PyTorch Dataset class
class JobPostingDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["description"]
        text = "No description provided." if not isinstance(text, str) or text.strip() == "" else text

        tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(int(self.df.iloc[idx]["fraudulent"]), dtype=torch.long)
        }

# Training function with MLflow tracking
def train_model():
    train_dataset = JobPostingDataset(df)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Smaller batch size for efficiency

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    optimizer = Adam(model.parameters(), lr=2e-5)  # Optimized learning rate

    scaler = GradScaler()  # For FP16 mixed precision training

    # Start MLflow Run
    with mlflow.start_run():
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_param("batch_size", 16)
        mlflow.log_param("learning_rate", 2e-5)
        mlflow.log_param("epochs", 1)

        model.train()
        num_epochs = 1
        total_loss = 0
        predictions, true_labels = [], []

        start_time = time.time()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                with autocast():  # Enable FP16 precision for speed
                    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                    labels=batch["label"])
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                logits = outputs.logits.detach().cpu().numpy()
                predictions.extend(np.argmax(logits, axis=1))
                true_labels.extend(batch["label"].cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(train_dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        training_time = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Time: {training_time:.2f}s")

        # Log metrics in MLflow
        mlflow.log_metric("loss", avg_loss)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("training_time", training_time)

        # Save model if it's the best so far
        os.makedirs("models", exist_ok=True)
        model_path = "models/fake_job_detector"
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        mlflow.pytorch.log_model(model, "model")
        mlflow.register_model("runs:/" + mlflow.active_run().info.run_id + "/model", "fake_job_detector")  # Register model
        print(f"Model training completed and saved to {model_path}")

if __name__ == "__main__":
    train_model()