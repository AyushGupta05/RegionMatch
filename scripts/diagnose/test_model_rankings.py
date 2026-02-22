import joblib
import pandas as pd

df = pd.read_csv("training_data_v1.csv")
pipe = joblib.load("location_model.joblib")
features = joblib.load("model_features.joblib")

df["model_score"] = pipe.predict(df[features])

top = df.sort_values("model_score", ascending=False)[["lad_code","lad_name","model_score"]].head(15)
bottom = df.sort_values("model_score", ascending=True)[["lad_code","lad_name","model_score"]].head(15)

print("\n=== TOP 15 LOCATIONS ===")
print(top.to_string(index=False))

print("\n=== BOTTOM 15 LOCATIONS ===")
print(bottom.to_string(index=False))

top.to_csv("top15_locations.csv", index=False)
bottom.to_csv("bottom15_locations.csv", index=False)
print("\nWrote top15_locations.csv and bottom15_locations.csv")