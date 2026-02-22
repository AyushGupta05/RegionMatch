import pandas as pd
import numpy as np

FILE = "training_data_clean.csv"
df = pd.read_csv(FILE)

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def minmax(x):
    x = to_num(x)
    if x.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.min()) / (x.max() - x.min())

# Core components (auto-skip if missing)
components = {}

if "approval_rate" in df.columns:
    components["planning_approval"] = minmax(df["approval_rate"])

if "median_decision_days" in df.columns:
    # lower days = better
    components["planning_speed"] = 1 - minmax(df["median_decision_days"])

if "job_liquidity_score_1_10" in df.columns:
    components["labour_liquidity"] = minmax(df["job_liquidity_score_1_10"])

if "scaling_index" in df.columns:
    components["scaling"] = minmax(df["scaling_index"])

# Ecosystem strength: pick best available density for your chosen "default"
ecosystem_cols = [c for c in [
    "tech_business_density",
    "core_tech_density",
    "innovation_density",
    "business_density",
    "tech_density"
] if c in df.columns]

if ecosystem_cols:
    # use the max of normalised ecosystem signals to avoid double-counting
    eco_norm = [minmax(df[c]) for c in ecosystem_cols]
    components["ecosystem"] = pd.concat(eco_norm, axis=1).max(axis=1)

if not components:
    raise SystemExit("No suitable columns found to build a target_score.")

# Weights (sum to 1.0). Adjust later if you want.
weights = {
    "planning_approval": 0.30,
    "planning_speed": 0.25,
    "labour_liquidity": 0.20,
    "scaling": 0.15,
    "ecosystem": 0.10,
}

# Renormalize weights to only included components
present = [k for k in weights if k in components]
wsum = sum(weights[k] for k in present)
for k in present:
    weights[k] /= wsum

score = np.zeros(len(df))
for k in present:
    score += weights[k] * components[k].to_numpy()

df["target_score"] = (100 * score).round(4)

# Fill remaining NaNs in numeric columns for training stability
id_cols = [c for c in ["lad_code","lad_name","council_id"] if c in df.columns]
num_cols = [c for c in df.columns if c not in id_cols]
for c in num_cols:
    df[c] = to_num(df[c])
df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

df.to_csv(FILE, index=False)

print("Created target_score with components:", present)
print("target_score stats:", float(df["target_score"].min()), float(df["target_score"].mean()), float(df["target_score"].max()))
