from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "models")

# Pick the latest checkpoint automatically
checkpoints = [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint")]
latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]

checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
export_path = os.path.join(CHECKPOINT_DIR, "aura_emotion_model")

print("📦 Exporting from:", checkpoint_path)
print("📁 Saving final model to:", export_path)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

tokenizer.save_pretrained(export_path)
model.save_pretrained(export_path)

print("✅ AURA model exported successfully!")
