import pandas as pd
import numpy as np

# INPUTS
BASE_TABLE = "training_table_canonical.csv"   # your LAD + Ibex table
SENT_FILE  = "city_sentiment_fixed.csv"

df = pd.read_csv(BASE_TABLE)
sent = pd.read_csv(SENT_FILE)

# Rename columns to ML-friendly names
sent = sent.rename(columns={
    "City": "city_name",
    "Job Liquidity Score (1-10)": "job_liquidity_score_1_10",
    "Reddit Sentiment Score (1-10)": "reddit_sentiment_score_1_10"
})

def clean_name(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("&", "and")
    while "  " in s:
        s = s.replace("  ", " ")
    # strip common UK LA suffix variants
    for suf in [
        ", city of", " city of",
        " city council", " borough council", " district council",
        " county council", " unitary authority",
        " metropolitan borough council", " metropolitan borough",
        " london borough council", " london borough",
        " council"
    ]:
        s = s.replace(suf, "")
    # normalize punctuation
    s = s.replace(".", "")
    return s

# Clean keys
df["name_clean"] = df["lad_name"].apply(clean_name)
sent["name_clean"] = sent["city_name"].apply(clean_name)

# If duplicates exist in sentiment data, average them
sent_agg = sent.groupby("name_clean", as_index=False).agg({
    "job_liquidity_score_1_10": "mean",
    "reddit_sentiment_score_1_10": "mean",
    "city_name": "first"
})

# Merge (left join keeps all LADs)
out = df.merge(
    sent_agg[["name_clean", "job_liquidity_score_1_10", "reddit_sentiment_score_1_10"]],
    on="name_clean",
    how="left"
).drop(columns=["name_clean"])

out.to_csv("training_table_with_sentiment.csv", index=False)

print("Saved: training_table_with_sentiment.csv")
print("Job liquidity coverage:", 1 - out["job_liquidity_score_1_10"].isna().mean())
print("Reddit sentiment coverage:", 1 - out["reddit_sentiment_score_1_10"].isna().mean())

# Quick stats
print("\nJob liquidity describe:")
print(out["job_liquidity_score_1_10"].describe())
print("\nReddit sentiment describe:")
print(out["reddit_sentiment_score_1_10"].describe())