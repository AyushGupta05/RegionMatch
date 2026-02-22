import pandas as pd
import numpy as np

IN = "training_data_v1.csv"
OUT = "training_data_clean.csv"

df = pd.read_csv(IN)

# Drop broken column
if "earnings_median" in df.columns:
    df = df.drop(columns=["earnings_median"])
    print("Dropped earnings_median")

# Coerce numeric columns (except identifiers)
id_cols = [c for c in ["lad_code","lad_name","council_id"] if c in df.columns]
for c in df.columns:
    if c in id_cols:
        continue
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Fill missing numeric columns with median
num_cols = [c for c in df.columns if c not in id_cols]
df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

df.to_csv(OUT, index=False)
print("Wrote", OUT, "shape:", df.shape)

# Report remaining missing
print("Remaining NaN cells:", int(df.isna().sum().sum()))
