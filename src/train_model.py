import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -----------------------------
# 1. Load cleaned dataset
# -----------------------------

DATA_PATH = os.path.join("..", "data", "processed", "aura_cleaned.csv")

df = pd.read_csv(DATA_PATH)

# Drop missing or empty text
df = df.dropna(subset=["clean_text", "aura_emotion"])
df["clean_text"] = df["clean_text"].astype(str)
df = df[df["clean_text"].str.strip() != ""]

print(f"✅ Dataset loaded: {len(df)} rows")

# -----------------------------
# 2. Encode emotion labels
# -----------------------------

label_map = {label: i for i, label in enumerate(sorted(df["aura_emotion"].unique()))}
id_to_label = {v: k for k, v in label_map.items()}

df["label_id"] = df["aura_emotion"].map(label_map)

print("✅ Label mapping:", label_map)

# -----------------------------
# 3. Train / validation split
# -----------------------------

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["clean_text"].tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"]
)

# -----------------------------
# 4. Load tokenizer & model
# -----------------------------

MODEL_NAME = "distilbert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_map),
    id2label=id_to_label,
    label2id=label_map
)

# -----------------------------
# 5. Tokenization (FIXED)
# -----------------------------

def tokenize(texts):
    texts = [str(t) for t in texts if isinstance(t, str) and t.strip() != ""]
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128
    )

train_encodings = tokenize(train_texts)
val_encodings = tokenize(val_texts)

# -----------------------------
# 6. Torch Dataset class
# -----------------------------

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)

# -----------------------------
# 7. Training configuration
# -----------------------------

training_args = TrainingArguments(
    output_dir="../models",
    learning_rate=2e-5,         
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=50
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# -----------------------------
# 8. Train model
# -----------------------------

print("🚀 Training AURA emotion model...")
trainer.train()

# -----------------------------
# 9. Save model
# -----------------------------

SAVE_PATH = "../models/aura_emotion_model"

model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print(f"✅ Model saved to {SAVE_PATH}")
