import pandas as pd
import numpy as np

print("=== LOADING FILES ===")

lad_path = "final_business_relocation_training_data.csv"
ibex_path = "ibex_features_by_council.csv"

lad = pd.read_csv(lad_path)
ibex = pd.read_csv(ibex_path)

print("LAD shape:", lad.shape)
print("Ibex shape:", ibex.shape)

# ----------------------------
# 1) DUPLICATES CHECK (LAD)
# ----------------------------
print("\n=== DUPLICATES CHECK (LAD) ===")
if "lad_code" in lad.columns:
    dup_counts = lad["lad_code"].value_counts()
    print("Unique LADs:", dup_counts.shape[0])
    print("Rows per LAD - top 10:")
    print(dup_counts.head(10))
else:
    print("No lad_code column found.")

# ----------------------------
# 2) IBEX MISSINGNESS + BASIC STATS
# ----------------------------
print("\n=== IBEX BASIC STATS ===")
num_cols = ibex.select_dtypes(include=[np.number]).columns.tolist()
print("Numeric columns:", num_cols)

print("\nMissingness (fraction missing) for Ibex numeric cols:")
print((ibex[num_cols].isna().mean()).sort_values(ascending=False))

print("\nDescribe Ibex numeric cols:")
print(ibex[num_cols].describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).T)

# ----------------------------
# 3) OUTLIERS (TOP/BOTTOM) FOR KEY IBEX FIELDS
# ----------------------------
key_fields = [c for c in ["approval_rate","apps_total","commercial_apps","median_decision_days","apps_decided"] if c in ibex.columns]

print("\n=== IBEX OUTLIERS (TOP/BOTTOM 10) ===")
for c in key_fields:
    print(f"\n--- {c} TOP 10 ---")
    print(ibex.sort_values(c, ascending=False)[["council_id","council_name",c]].head(10))
    print(f"--- {c} BOTTOM 10 ---")
    print(ibex.sort_values(c, ascending=True)[["council_id","council_name",c]].head(10))

# ----------------------------
# 4) QUICK CORRELATION (IBEX ONLY)
# ----------------------------
print("\n=== IBEX CORRELATION (numeric) ===")
if len(num_cols) >= 2:
    corr = ibex[num_cols].corr(numeric_only=True)
    print(corr)
else:
    print("Not enough numeric cols for correlation.")

# ----------------------------
# 5) OPTIONAL: IF YOU ALREADY HAVE A MERGED FILE, CHECK IT TOO
# ----------------------------
merged_path = "training_data_with_ibex.csv"
try:
    merged = pd.read_csv(merged_path)
    print("\n=== MERGED FILE CHECK ===")
    print("Merged shape:", merged.shape)

    merged_ibex_cols = [c for c in ["council_id","approval_rate","apps_total","commercial_apps","median_decision_days"] if c in merged.columns]

    if merged_ibex_cols:
        print("\nMerged missingness for Ibex columns:")
        print(merged[merged_ibex_cols].isna().mean().sort_values(ascending=False))

        if "council_id" in merged.columns:
            match_rate = 1 - merged["council_id"].isna().mean()
            print("\nIbex match rate (rows with council_id):", match_rate)

    else:
        print("Merged file exists but no Ibex columns found.")
except FileNotFoundError:
    print("\n(No merged file found yet: training_data_with_ibex.csv)")

print("\n=== DONE ===")
