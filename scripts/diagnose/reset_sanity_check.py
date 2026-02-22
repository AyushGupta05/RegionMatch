import pandas as pd, joblib, numpy as np

df = pd.read_csv("training_data_clean.csv")
features = joblib.load("model_features.joblib")
model = joblib.load("location_model.joblib")

pred = model.predict(df[features])

print("Pred min/mean/max:", float(pred.min()), float(pred.mean()), float(pred.max()))
print("Pred std:", float(np.std(pred)))

# show top 10 LADs by prediction
out = df[["lad_code","lad_name"]].copy()
out["pred"] = pred
print(out.sort_values("pred", ascending=False).head(10).to_string(index=False))
