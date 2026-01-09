import pandas as pd
df = pd.read_csv("data/processed/aura_cleaned.csv")
print(df[["clean_text", "aura_emotion"]].head())
