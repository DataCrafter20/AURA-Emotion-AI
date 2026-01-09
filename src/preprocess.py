import pandas as pd
import re

INPUT_PATH = "data/raw/go_emotions.csv"
OUTPUT_PATH = "data/processed/aura_cleaned.csv"

AURA_MAP = {
    "sadness": "sadness",
    "grief": "sadness",
    "disappointment": "sadness",
    "fear": "anxiety",
    "nervousness": "anxiety",
    "anger": "stress",
    "frustration": "stress",
    "annoyance": "stress",
    "exhaustion": "exhaustion",
    "neutral": "neutral"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()



def main():
    df = pd.read_csv(INPUT_PATH)

    # Identify emotion columns (exclude text)
    emotion_columns = [col for col in df.columns if col != "text"]

    def extract_emotion(row):
        for emotion in emotion_columns:
            if row[emotion] == 1:
                return emotion
        return None

    df["raw_emotion"] = df.apply(extract_emotion, axis=1)

    df["clean_text"] = df["text"].apply(clean_text)
    df["aura_emotion"] = df["raw_emotion"].map(AURA_MAP)

    df.dropna(subset=["aura_emotion"], inplace=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print("✅ Cleaned data saved to:", OUTPUT_PATH)



if __name__ == "__main__":
    main()
