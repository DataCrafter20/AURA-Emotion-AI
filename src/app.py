import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from responses import generate_response

# ---------------------------
# Load model & tokenizer once
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "aura_emotion_model")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

ID_TO_EMOTION = {
    0: "anxiety",
    1: "neutral",
    2: "sadness",
    3: "stress"
}

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()

    emotion = ID_TO_EMOTION[pred_id]
    confidence = probs[0][pred_id].item()
    response = generate_response(emotion)

    return emotion, confidence, response

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AURA – Emotion-Aware Assistant", page_icon="🤖")

st.title("🤖 AURA")
st.subheader("Emotion-Aware AI Assistant")

user_input = st.text_area(
    "How are you feeling today?",
    placeholder="Type how you feel..."
)

if st.button("Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        emotion, confidence, response = predict(user_input)

        st.markdown(f"### 🧠 Detected Emotion: **{emotion.capitalize()}**")
        st.markdown(f"**Confidence:** {confidence:.2f}")
        st.markdown("---")
        st.markdown(f"### 💬 AURA :")
        st.info(response)
