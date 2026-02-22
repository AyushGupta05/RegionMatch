import pandas as pd

df = pd.read_csv("training_data_clean.csv")
cent = pd.read_csv("lad_centroids.csv")

out = df.merge(cent, on="lad_code", how="left")

print("Missing centroid rate:", out["lad_lat"].isna().mean())
out.to_csv("training_data_geo.csv", index=False)
print("Saved training_data_geo.csv", out.shape)
