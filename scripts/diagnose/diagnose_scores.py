import joblib, pandas as pd, numpy as np

df = pd.read_csv("training_data_v1.csv")
pipe = joblib.load("location_model.joblib")
features = joblib.load("model_features.joblib")

base = pipe.predict(df[features])

print("Base score stats:")
print("  mean:", float(np.mean(base)))
print("  std :", float(np.std(base)))
print("  min :", float(np.min(base)))
print("  max :", float(np.max(base)))

# If you have any scenario adjustment columns you use later, inspect them
for col in ["approval_rate","median_decision_days","job_liquidity_score_1_10","earnings_median","business_density"]:
    if col in df.columns:
        x = df[col].to_numpy()
        x = x[np.isfinite(x)]
        if len(x):
            print(f"{col} std:", float(np.std(x)))
