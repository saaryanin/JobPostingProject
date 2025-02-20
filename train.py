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
from torch.cuda.amp import autocast, GradScaler  # Mixed precision training
import time

# ✅ MLflow Experiment Setup
mlflow.set_experiment("Fake Job Detector Experiment")

# ✅ Device Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ Load Dataset (Limit to 5000 samples for quick training)
df = pd.read_csv("datasets/fake_job_postings.csv")
df = df.sample(n=5000, random_state=42).reset_index(drop=True)
print(f"Dataset Size: {len(df)} samples")

# ✅ Model Configurations (Train Two Different Models)
model_variants = [
    {"name": "bert-base-uncased", "batch_size": 16, "learning_rate": 2e-5},
    {"name": "bert-large-uncased", "batch_size": 8, "learning_rate": 3e-5}
]

# ✅ Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Same tokenizer for both

# ✅ Custom PyTorch Dataset
class JobPostingDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["description"]
        text = "No description provided." if not isinstance(text, str) or text.strip() == "" else text

        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(0).to(torch.int64),
            "attention_mask": tokens["attention_mask"].squeeze(0).to(torch.int64),
            "label": torch.tensor(int(self.df.iloc[idx]["fraudulent"]), dtype=torch.long)
        }

# ✅ Training Function
def train_and_evaluate_model(model_name, batch_size, learning_rate):
    print(f"\n🔹 Training Model: {model_name} | Batch Size: {batch_size} | LR: {learning_rate}")

    train_dataset = JobPostingDataset(df)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    scaler = GradScaler()  # Mixed Precision

    best_accuracy = 0.0
    best_model_path = None

    # ✅ MLflow Run
    with mlflow.start_run():
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", 2)

        model.train()
        num_epochs = 2
        total_loss = 0
        predictions, true_labels = [], []

        start_time = time.time()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                with autocast():  # ✅ Enable FP16 Precision
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["label"]
                    )
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                logits = outputs.logits.detach().cpu().numpy()
                predictions.extend(np.argmax(logits, axis=1))
                true_labels.extend(batch["label"].cpu().numpy())

        # ✅ Compute Metrics
        avg_loss = total_loss / len(train_dataloader)
        accuracy = accuracy_score(true_labels, predictions)
        training_time = time.time() - start_time

        print(f"✅ Model: {model_name} | Accuracy: {accuracy:.4f} | Loss: {avg_loss:.4f} | Time: {training_time:.2f}s")

        # ✅ Log Metrics
        mlflow.log_metric("loss", avg_loss)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("training_time", training_time)

        # ✅ Save Model if Accuracy is the Best
        model_path = f"models/{model_name.replace('-', '_')}"
        os.makedirs(model_path, exist_ok=True)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = model_path
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            mlflow.pytorch.log_model(model, "model")
            print(f"✔️ New Best Model Saved: {model_path}")

    return best_model_path, best_accuracy

# ✅ Train Both Models and Select the Best One
best_overall_model = None
best_overall_accuracy = 0.0

for model_config in model_variants:
    model_path, accuracy = train_and_evaluate_model(
        model_config["name"],
        model_config["batch_size"],
        model_config["learning_rate"]
    )

    # ✅ Update Best Model
    if accuracy > best_overall_accuracy:
        best_overall_accuracy = accuracy
        best_overall_model = model_path

print(f"\n🏆 Best Model Selected: {best_overall_model} | Accuracy: {best_overall_accuracy:.4f}")

# ✅ Save the Best Model Name for Future Use
with open("models/best_model.txt", "w") as f:
    f.write(best_overall_model)
print(f"📌 Best Model Path Saved: models/best_model.txt")

