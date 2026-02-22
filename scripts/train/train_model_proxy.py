import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Prefer geo/clean dataset if available
DATA_PATH = "training_data_geo.csv" if os.path.exists("training_data_geo.csv") else "training_data_clean.csv"

def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def minmax_safe(s: pd.Series, fill_value: float) -> pd.Series:
    """Min-max scale with guaranteed finite output."""
    x = to_num(s).copy()
    x = x.fillna(fill_value)

    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or abs(hi - lo) < 1e-12:
        return pd.Series(np.zeros(len(x)), index=x.index)

    return (x - lo) / (hi - lo)

# -----------------------------
# 1) Load
# -----------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH, "shape:", df.shape)

id_cols = [c for c in ["lad_code", "lad_name", "council_id", "lad_lat", "lad_lng"] if c in df.columns]
for c in ["lad_code", "lad_name"]:
    if c not in df.columns:
        raise SystemExit(f"Missing required column: {c}")

# -----------------------------
# 2) Coerce features to numeric (exclude ids)
# -----------------------------
candidate_cols = [c for c in df.columns if c not in id_cols]

for c in candidate_cols:
    df[c] = to_num(df[c])

# Drop columns that are entirely NaN or constant
numeric_cols = []
dropped_const = []
dropped_allnan = []
for c in candidate_cols:
    col = df[c]
    if col.isna().all():
        dropped_allnan.append(c)
        continue
    if col.nunique(dropna=True) <= 1:
        dropped_const.append(c)
        continue
    numeric_cols.append(c)

X = df[numeric_cols].copy()

print("Numeric feature cols:", len(numeric_cols))
if dropped_allnan:
    print("Dropped all-NaN cols:", len(dropped_allnan))
if dropped_const:
    print("Dropped constant cols:", len(dropped_const))

# -----------------------------
# 3) Proxy target (stable + more meaningful)
# -----------------------------
# Fill values chosen to be neutral/median-like
approval_fill = float(to_num(df["approval_rate"]).median()) if "approval_rate" in df.columns else 0.85
delay_fill    = float(to_num(df["median_decision_days"]).median()) if "median_decision_days" in df.columns else 60.0
liq_fill      = float(to_num(df["job_liquidity_score_1_10"]).median()) if "job_liquidity_score_1_10" in df.columns else 5.0
sent_fill     = float(to_num(df["reddit_sentiment_score_1_10"]).median()) if "reddit_sentiment_score_1_10" in df.columns else 5.0

# Ecosystem / business environment fills
scale_fill    = float(to_num(df["scaling_index"]).median()) if "scaling_index" in df.columns else 0.0
eco_fill      = float(to_num(df["tech_business_density"]).median()) if "tech_business_density" in df.columns else 0.0

approval = minmax_safe(df["approval_rate"], approval_fill) if "approval_rate" in df.columns else 0
delay    = minmax_safe(df["median_decision_days"], delay_fill) if "median_decision_days" in df.columns else 0
liq      = minmax_safe(df["job_liquidity_score_1_10"], liq_fill) if "job_liquidity_score_1_10" in df.columns else 0
sent     = minmax_safe(df["reddit_sentiment_score_1_10"], sent_fill) if "reddit_sentiment_score_1_10" in df.columns else 0
scale    = minmax_safe(df["scaling_index"], scale_fill) if "scaling_index" in df.columns else 0
eco      = minmax_safe(df["tech_business_density"], eco_fill) if "tech_business_density" in df.columns else 0

# NOTE: delay is "bad", so subtract it.
# Removed the penalty on business_density because it can dominate rankings.
y = (
    0.32 * approval
    + 0.22 * (1 - delay)
    + 0.20 * liq
    + 0.10 * sent
    + 0.10 * scale
    + 0.06 * eco
).astype(float)

# Final guard: y must be finite
if not np.isfinite(y).all():
    bad = np.where(~np.isfinite(y))[0][:10]
    raise SystemExit(f"Proxy target contains non-finite values at rows: {bad}")

print("Proxy target y stats:",
      "min", float(np.min(y)),
      "mean", float(np.mean(y)),
      "max", float(np.max(y)),
      "std", float(np.std(y)))

# -----------------------------
# 4) Model pipeline
# -----------------------------
pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0, random_state=42))
])

# -----------------------------
# 5) Cross-validation
# -----------------------------
cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")

print("5-fold CV R^2 scores:", scores)
print("Mean R^2:", float(scores.mean()), "Std:", float(scores.std()))

# -----------------------------
# 6) Fit final model and save
# -----------------------------
pipe.fit(X, y)

joblib.dump(pipe, "location_model.joblib")
joblib.dump(numeric_cols, "model_features.joblib")

print("Saved: location_model.joblib")
print("Saved: model_features.joblib")
print("Num features:", len(numeric_cols))