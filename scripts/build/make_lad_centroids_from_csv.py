import pandas as pd

src = r"data/lad_lookup.csv"
df = pd.read_csv(src, encoding="utf-8", low_memory=False)

code_col = None
for c in ["LAD23CD","LAD22CD","LAD21CD","LADCD","lad_code"]:
    if c in df.columns:
        code_col = c
        break

lat_col = None
for c in ["LAT","Lat","lat","LATITUDE","Latitude"]:
    if c in df.columns:
        lat_col = c
        break

lng_col = None
for c in ["LONG","LON","Long","lon","lng","LONGITUDE","Longitude"]:
    if c in df.columns:
        lng_col = c
        break

if code_col is None or lat_col is None or lng_col is None:
    raise SystemExit(f"Missing expected columns. Found columns include: {list(df.columns)[:40]}")

out = pd.DataFrame({
    "lad_code": df[code_col].astype(str),
    "lad_lat": pd.to_numeric(df[lat_col], errors="coerce"),
    "lad_lng": pd.to_numeric(df[lng_col], errors="coerce"),
}).dropna(subset=["lad_lat","lad_lng"]).drop_duplicates("lad_code")

out.to_csv("lad_centroids.csv", index=False)
print("Saved lad_centroids.csv rows:", len(out))
print(out.head(5).to_string(index=False))
