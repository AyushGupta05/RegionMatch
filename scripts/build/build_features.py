import pandas as pd, joblib

df = pd.read_csv("training_data_clean.csv")

# Identify non-features
id_cols = [c for c in ["lad_code","lad_name","council_id"] if c in df.columns]

# Detect target column (YOU MUST CONFIRM THIS)
candidate_targets = [c for c in ["target","target_score","y","label","score_raw"] if c in df.columns]
target = candidate_targets[0] if candidate_targets else None
print("Detected target:", target)

features = [c for c in df.columns if c not in id_cols and c != target]

joblib.dump(features, "model_features.joblib")
print("Saved model_features.joblib with", len(features), "features")
print("First 20 features:", features[:20])
