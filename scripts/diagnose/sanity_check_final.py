import pandas as pd
import numpy as np

df = pd.read_csv("training_table_with_sentiment.csv")

print("\n=== BASIC SHAPE ===")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Unique LADs:", df["lad_code"].nunique())

print("\n=== DUPLICATE CHECK ===")
dupes = df["lad_code"].duplicated().sum()
print("Duplicate LAD rows:", dupes)

print("\n=== MISSINGNESS (TOP FEATURES) ===")
cols = [
    "approval_rate",
    "apps_total",
    "commercial_apps",
    "median_decision_days",
    "job_liquidity_score_1_10",
    "reddit_sentiment_score_1_10"
]
present = [c for c in cols if c in df.columns]
print(df[present].isna().mean().sort_values(ascending=False))

print("\n=== VALUE RANGES ===")
if "approval_rate" in df:
    print("approval_rate min/max:", df["approval_rate"].min(), df["approval_rate"].max())
if "job_liquidity_score_1_10" in df:
    print("job_liquidity min/max:", df["job_liquidity_score_1_10"].min(), df["job_liquidity_score_1_10"].max())
if "reddit_sentiment_score_1_10" in df:
    print("reddit_sentiment min/max:", df["reddit_sentiment_score_1_10"].min(), df["reddit_sentiment_score_1_10"].max())

print("\n=== BASIC DISTRIBUTIONS ===")
for c in ["approval_rate","commercial_apps","median_decision_days"]:
    if c in df.columns:
        print(f"\n{c}")
        print(df[c].describe())