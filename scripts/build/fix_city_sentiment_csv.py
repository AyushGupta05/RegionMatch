import csv

inp = "city_sentiment.csv"
out = "city_sentiment_fixed.csv"

rows = []

with open(inp, "r", encoding="utf-8", newline="") as f:
    # Read raw lines
    lines = [line.rstrip("\n") for line in f if line.strip()]

header = lines[0]
rows.append(["City", "Job Liquidity Score (1-10)", "Reddit Sentiment Score (1-10)"])

for i, line in enumerate(lines[1:], start=2):
    # Split from the RIGHT because the last two values are always numbers
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3:
        raise SystemExit(f"Bad line {i}: {line}")

    # last two columns are the scores
    reddit = parts[-1]
    liquidity = parts[-2]
    city = ",".join(parts[:-2]).strip()

    rows.append([city, liquidity, reddit])

with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    w.writerows(rows)

print("Wrote:", out)
print("Rows:", len(rows) - 1)