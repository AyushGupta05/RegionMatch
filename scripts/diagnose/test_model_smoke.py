import joblib
import pandas as pd
import numpy as np

df = pd.read_csv("training_data_v1.csv")

pipe = joblib.load("location_model.joblib")
features = joblib.load("model_features.joblib")

X = df[features].copy()
pred = pipe.predict(X)

print("Pred shape:", pred.shape)
print("Pred min/mean/max:", float(np.min(pred)), float(np.mean(pred)), float(np.max(pred)))
print("Any NaN in predictions?:", np.isnan(pred).any())
print("First 5 predictions:", pred[:5])