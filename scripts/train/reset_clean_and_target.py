import pandas as pd
import numpy as np

IN = "training_data_v1.csv"
OUT = "training_data_clean.csv"

df = pd.read_csv(IN)
print("Loaded", IN, "shape:", df.shape)

# --------------------------
# Identify columns
# --------------------------
id_cols = [c for c in ["lad_code","lad_name","lad_lat","lad_lng","council_id"] if c in df.columns]

# Drop known broken / leakage / unusable features if present
drop_cols = []
for c in ["earnings_median"]:  # known constant from your diagnosis
    if c in df.columns:
        drop_cols.append(c)

if drop_cols:
    df = df.drop(columns=drop_cols)
    print("Dropped:", drop_cols)

# --------------------------
# Convert to numeric (non-id)
# --------------------------
for c in df.columns:
    if c in id_cols:
        continue
    df[c] = pd.to_numeric(df[c], errors="coerce")

# --------------------------
# Drop high-missing columns (>45% missing)
# (except we keep core business counts/densities even if sparse)
# --------------------------
protected = set([
    "approval_rate","median_decision_days","apps_total","apps_decided","commercial_apps",
    "job_liquidity_score_1_10",
    "total_businesses","business_density",
    "tech_business_total","tech_business_density",
    "core_tech_density","innovation_density","creative_density","business_services_density",
    "scaling_index","micro_ratio","sme_ratio","large_ratio"
])

miss = df.isna().mean()
to_drop = [c for c in df.columns if (miss[c] > 0.45 and c not in id_cols and c not in protected)]
if to_drop:
    df = df.drop(columns=to_drop)
    print("Dropped high-missing columns:", len(to_drop))

# --------------------------
# Fill missing numeric values with median
# --------------------------
num_cols = [c for c in df.columns if c not in id_cols]
df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

# --------------------------
# Drop constant numeric columns (<=1 unique value)
# --------------------------
const_drop = []
for c in num_cols:
    if df[c].nunique(dropna=True) <= 1:
        const_drop.append(c)
if const_drop:
    df = df.drop(columns=const_drop)
    print("Dropped constant columns:", len(const_drop))

# --------------------------
# Create target_score (0–100)
# (Explainable composite — works even without labels)
# --------------------------
def minmax(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

parts = {}

if "approval_rate" in df.columns:
    parts["approval"] = minmax(df["approval_rate"])

if "median_decision_days" in df.columns:
    parts["speed"] = 1 - minmax(df["median_decision_days"])

if "job_liquidity_score_1_10" in df.columns:
    parts["liquidity"] = minmax(df["job_liquidity_score_1_10"])

if "scaling_index" in df.columns:
    parts["scaling"] = minmax(df["scaling_index"])

eco_cols = [c for c in ["tech_business_density","core_tech_density","innovation_density","business_density","tech_density"] if c in df.columns]
if eco_cols:
    eco = pd.concat([minmax(df[c]) for c in eco_cols], axis=1).max(axis=1)
    parts["ecosystem"] = eco

if not parts:
    raise SystemExit("Cannot build target_score: none of the expected columns exist after cleaning.")

weights = {"approval":0.30,"speed":0.25,"liquidity":0.20,"scaling":0.15,"ecosystem":0.10}
present = [k for k in weights if k in parts]
wsum = sum(weights[k] for k in present)
weights = {k:weights[k]/wsum for k in present}

score = np.zeros(len(df))
for k in present:
    score += weights[k] * parts[k].to_numpy()

df["target_score"] = (100 * score).round(4)

# Fill again (in case target creation introduced NaNs)
num_cols2 = [c for c in df.columns if c not in id_cols]
df[num_cols2] = df[num_cols2].fillna(df[num_cols2].median(numeric_only=True))

df.to_csv(OUT, index=False)
print("Wrote", OUT, "shape:", df.shape)
print("target_score min/mean/max:", float(df["target_score"].min()), float(df["target_score"].mean()), float(df["target_score"].max()))
print("Columns:", len(df.columns))
