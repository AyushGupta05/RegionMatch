import pandas as pd, numpy as np, joblib

df = pd.read_csv("training_data_v1.csv")
print("Rows:", len(df), "Cols:", len(df.columns))

features = joblib.load("model_features.joblib")
print("Feature count:", len(features))

missing = [f for f in features if f not in df.columns]
print("Missing features:", missing[:20], "..." if len(missing)>20 else "")
print("Missing count:", len(missing))

# Check types and NaNs
bad = []
for f in features:
    if f in df.columns:
        x = pd.to_numeric(df[f], errors="coerce")
        nan_rate = float(x.isna().mean())
        uniq = int(x.nunique(dropna=True))
        if nan_rate > 0.5 or uniq <= 1:
            bad.append((f, nan_rate, uniq))
print("Bad/constant features (nan_rate>0.5 or uniq<=1):", len(bad))
print("Top 25 bad features:")
for row in bad[:25]:
    print(" ", row)

# Try model predict
model = joblib.load("location_model.joblib")
X = df[[f for f in features if f in df.columns]].copy()

# coerce numeric
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

print("X shape:", X.shape)
print("X total NaN cells:", int(X.isna().sum().sum()))

# Fill NaNs for prediction test
X2 = X.fillna(X.median(numeric_only=True))
pred = model.predict(X2)
print("Pred stats:", float(np.min(pred)), float(np.mean(pred)), float(np.max(pred)), "std:", float(np.std(pred)))
