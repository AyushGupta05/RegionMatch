import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DATA_PATH = "training_data_v1.csv"

def minmax_safe(s: pd.Series, fill_value: float) -> pd.Series:
    """Min-max scale with guaranteed finite output."""
    x = s.astype(float).copy()
    x = x.fillna(fill_value)

    lo = float(x.min())
    hi = float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        # If constant or invalid, return zeros (no signal)
        return pd.Series(np.zeros(len(x)), index=x.index)

    return (x - lo) / (hi - lo)

# -----------------------------
# 1) Load
# -----------------------------
df = pd.read_csv(DATA_PATH)




id_cols = ["lad_code", "lad_name"]
for c in id_cols:
    if c not in df.columns:
        raise SystemExit(f"Missing required column: {c}")

# -----------------------------
# 2) Features (numeric only)
# -----------------------------
numeric_cols = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]
X = df[numeric_cols].copy()

# -----------------------------
# 3) Proxy target (SAFE)
# -----------------------------
# Fill values chosen to be neutral/median-like
approval_fill = float(df["approval_rate"].median()) if "approval_rate" in df.columns else 0.88
delay_fill    = float(df["median_decision_days"].median()) if "median_decision_days" in df.columns else 55.0
liq_fill      = 5.0
sent_fill     = 5.0
density_fill  = float(df["business_density"].median()) if "business_density" in df.columns else 0.0

approval = minmax_safe(df["approval_rate"], approval_fill) if "approval_rate" in df.columns else 0
delay    = minmax_safe(df["median_decision_days"], delay_fill) if "median_decision_days" in df.columns else 0
liq      = minmax_safe(df["job_liquidity_score_1_10"], liq_fill) if "job_liquidity_score_1_10" in df.columns else 0
sent     = minmax_safe(df["reddit_sentiment_score_1_10"], sent_fill) if "reddit_sentiment_score_1_10" in df.columns else 0
density  = minmax_safe(df["business_density"], density_fill) if "business_density" in df.columns else 0

y = (
    0.35 * approval
    - 0.25 * delay
    + 0.20 * liq
    + 0.10 * sent
    - 0.10 * density
).astype(float)

# Final guard: y must be finite
if not np.isfinite(y).all():
    bad = np.where(~np.isfinite(y))[0][:10]
    raise SystemExit(f"Proxy target contains non-finite values at rows: {bad}")

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
print("Mean R^2:", scores.mean(), "Std:", scores.std())

# -----------------------------
# 6) Fit final model and save
# -----------------------------
pipe.fit(X, y)

joblib.dump(pipe, "location_model.joblib")
joblib.dump(numeric_cols, "model_features.joblib")

print("Saved: location_model.joblib")
print("Saved: model_features.joblib")
print("Num features:", len(numeric_cols))