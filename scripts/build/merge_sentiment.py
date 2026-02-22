import pandas as pd

df = pd.read_csv("training_table_plus_ibex.csv")
sent = pd.read_csv("city_sentiment_fixed.csv")

sent = sent.rename(columns={
    "City": "city_name",
    "Job Liquidity Score (1-10)": "job_liquidity_score_1_10",
    "Reddit Sentiment Score (1-10)": "reddit_sentiment_score_1_10"
})

def clean(s):
    s = str(s).lower().strip()
    s = s.replace("&", "and")
    while "  " in s:
        s = s.replace("  ", " ")
    for suf in [
        " city council", " borough council", " district council",
        " county council", " unitary authority",
        " metropolitan borough council", " metropolitan borough",
        " london borough council", " london borough",
        " council", ", city of", " city of"
    ]:
        s = s.replace(suf, "")
    s = s.replace(".", "")
    return s

df["key"] = df["lad_name"].apply(clean)
sent["key"] = sent["city_name"].apply(clean)

# If duplicates exist in sentiment, average them
sent = sent.groupby("key", as_index=False).agg({
    "job_liquidity_score_1_10": "mean",
    "reddit_sentiment_score_1_10": "mean"
})

out = df.merge(sent, on="key", how="left").drop(columns=["key"])
out.to_csv("training_table_final.csv", index=False)

print("Saved training_table_final.csv", out.shape)
print("Sentiment coverage:", 1 - out["job_liquidity_score_1_10"].isna().mean())
print("Ibex coverage:", 1 - out["approval_rate"].isna().mean())
