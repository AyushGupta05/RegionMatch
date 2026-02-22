import pandas as pd

lad = pd.read_csv("lad_one_row.csv")
ibex = pd.read_csv("ibex_features_by_council.csv")

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
        " council", ", city of", " city of"
    ]:
        s = s.replace(suf, "")
    s = s.replace(".", "")
    return s

lad["key"] = lad["lad_name"].apply(clean)
ibex["key"] = ibex["council_name"].fillna("").apply(clean)

merged = lad.merge(
    ibex.drop(columns=["council_name"]),
    on="key",
    how="left"
).drop(columns=["key"])

merged.to_csv("training_table_plus_ibex.csv", index=False)

print("Saved training_table_plus_ibex.csv", merged.shape)
print("Ibex match rate:", 1 - merged["approval_rate"].isna().mean())
