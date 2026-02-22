import pandas as pd, joblib

df = pd.read_csv("training_data_clean.csv")

id_cols = set([c for c in ["lad_code","lad_name","lad_lat","lad_lng","council_id"] if c in df.columns])
target = "target_score"

features = [c for c in df.columns if c not in id_cols and c != target]

joblib.dump(features, "model_features.joblib")

print("Saved model_features.joblib")
print("Feature count:", len(features))
print("First 25:", features[:25])
print("Contains council_id?", "council_id" in features)
