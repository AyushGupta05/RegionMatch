import pandas as pd

lad = pd.read_csv("final_business_relocation_training_data.csv")

# aggregate earnings (since you have 3 rows per LAD)
earn = lad.groupby("lad_code")["median_weekly_earnings"].agg(
    earnings_min="min",
    earnings_median="median",
    earnings_max="max"
).reset_index()

lad_base = lad.drop_duplicates(subset=["lad_code"]).copy()
lad_base = lad_base.drop(columns=["median_weekly_earnings"]).merge(earn, on="lad_code", how="left")

lad_base.to_csv("lad_one_row.csv", index=False)
print("Saved lad_one_row.csv", lad_base.shape)
