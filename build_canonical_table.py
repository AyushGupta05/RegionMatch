import pandas as pd

lad = pd.read_csv("final_business_relocation_training_data.csv")
ibex = pd.read_csv("ibex_features_by_council.csv")

# --- Deduplicate LAD to one row ---
earn = lad.groupby("lad_code")["median_weekly_earnings"].agg(
    earnings_min="min",
    earnings_median="median",
    earnings_max="max"
).reset_index()

lad_base = lad.drop_duplicates(subset=["lad_code"]).copy()
lad_base = lad_base.drop(columns=["median_weekly_earnings"]).merge(
    earn, on="lad_code", how="left"
)

# --- Clean names for joining ---
def clean(s):
    s = str(s).lower().strip()
    s = s.replace("&", "and")
    while "  " in s:
        s = s.replace("  ", " ")
    for suf in [
        " city council", " borough council", " district council",
        " county council", " unitary authority",
        " metropolitan borough council", " metropolitan borough",
        " london borough council", " london borough",
        " council"
    ]:
        s = s.replace(suf, "")
    return s

lad_base["name_clean"] = lad_base["lad_name"].apply(clean)
ibex["name_clean"] = ibex["council_name"].fillna("").apply(clean)

# --- Merge Ibex into LAD ---
merged = lad_base.merge(
    ibex.drop(columns=["council_name"]),
    on="name_clean",
    how="left"
).drop(columns=["name_clean"])

# --- Save canonical table ---
merged.to_csv("training_table_canonical.csv", index=False)

print("Saved training_table_canonical.csv")
print("Rows:", len(merged))
print("Unique LADs:", merged["lad_code"].nunique())
print("Ibex match rate:", 1 - merged["council_id"].isna().mean())
