import os
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="ReloCopilot",
    page_icon="ðŸ§­",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# THEME / STYLE
# ============================================================
st.markdown("""
<style>
.stApp {
  background: radial-gradient(1200px 800px at 20% 10%, rgba(56,189,248,0.18), transparent 50%),
              radial-gradient(900px 600px at 90% 20%, rgba(59,130,246,0.16), transparent 40%),
              linear-gradient(180deg, #020617, #020617);
  color: #e2e8f0;
}
[data-testid="stSidebar"] > div:first-child {
  background: linear-gradient(180deg, rgba(2,6,23,0.96), rgba(2,6,23,0.92));
  border-right: 1px solid rgba(255,255,255,0.08);
}
.block-container { padding-top: 0.6rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CITY DATA (Expanded list)
# ============================================================
UK_CITIES = [
    # England
    ("London", -0.1276, 51.5072),
    ("Birmingham", -1.8904, 52.4862),
    ("Manchester", -2.2426, 53.4808),
    ("Leeds", -1.5491, 53.8008),
    ("Liverpool", -2.9916, 53.4084),
    ("Bristol", -2.5879, 51.4545),
    ("Sheffield", -1.4701, 53.3811),
    ("Newcastle upon Tyne", -1.6178, 54.9783),
    ("Nottingham", -1.1505, 52.9548),
    ("Leicester", -1.1332, 52.6369),
    ("Southampton", -1.4043, 50.9097),
    ("Portsmouth", -1.0873, 50.8198),
    ("Brighton", -0.1364, 50.8225),
    ("Cambridge", 0.1218, 52.2053),
    ("Oxford", -1.2577, 51.7520),
    ("Reading", -0.9781, 51.4543),
    ("Milton Keynes", -0.7594, 52.0406),
    ("Luton", -0.4176, 51.8797),
    ("Peterborough", -0.2420, 52.5695),
    ("Norwich", 1.2974, 52.6309),
    ("Ipswich", 1.1555, 52.0567),
    ("York", -1.0815, 53.9590),
    ("Hull", -0.3367, 53.7457),
    ("Middlesbrough", -1.2348, 54.5742),
    ("Sunderland", -1.3822, 54.9069),
    ("Derby", -1.4766, 52.9225),
    ("Stoke-on-Trent", -2.1794, 53.0027),
    ("Wolverhampton", -2.1276, 52.5862),
    ("Coventry", -1.5106, 52.4068),
    ("Northampton", -0.8901, 52.2405),
    ("Cheltenham", -2.0713, 51.8994),
    ("Swindon", -1.7809, 51.5558),
    ("Exeter", -3.5339, 50.7184),
    ("Plymouth", -4.1427, 50.3755),
    ("Bournemouth", -1.8795, 50.7192),

    # Wales
    ("Cardiff", -3.1791, 51.4816),
    ("Swansea", -3.9436, 51.6214),
    ("Newport", -2.9984, 51.5842),

    # Scotland
    ("Edinburgh", -3.1883, 55.9533),
    ("Glasgow", -4.2518, 55.8642),
    ("Aberdeen", -2.0943, 57.1497),
    ("Dundee", -2.9707, 56.4620),
    ("Inverness", -4.2247, 57.4778),

    # Northern Ireland
    ("Belfast", -5.9301, 54.5973),
    ("Derry/Londonderry", -7.3092, 54.9966),
]
cities_df = pd.DataFrame(UK_CITIES, columns=["city", "lng", "lat"])

INDUSTRIES = [
    "Technology", "Creative", "Innovation",
    "Business Services", "Retail/Hospitality", "Industrial/Logistics"
]
URGENCY = ["<3 months", "3-6 months", "6+ months"]

# ============================================================
# MAP CONFIG
# ============================================================
MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN", "")

# ============================================================
# HELPERS
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def clamp_to_uk(lng, lat):
    return clamp(lng, -8.8, 2.3), clamp(lat, 49.8, 60.9)

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance (km)."""
    R = 6371.0
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(float(lat2))
    lon2 = np.radians(float(lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def minmax_series(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

def safe_dataset_path():
    """Prefer geo -> clean -> v1, checking repo and processed folders."""
    candidates = [
        REPO_ROOT / "data" / "processed" / "training_data_geo.csv",
        REPO_ROOT / "training_data_geo.csv",
        REPO_ROOT / "data" / "processed" / "training_data_clean.csv",
        REPO_ROOT / "training_data_clean.csv",
        REPO_ROOT / "training_data_v1.csv",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("No training dataset found in expected locations.")


REPO_ROOT = Path(__file__).resolve().parents[1]

# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load(REPO_ROOT / "models" / "location_model.joblib")
    features = joblib.load(REPO_ROOT / "models" / "model_features.joblib")
    return model, features

@st.cache_data
def load_data(path: str):
    return pd.read_csv(path)

pipe, feature_list = load_model()
DATA_PATH = safe_dataset_path()
df = load_data(DATA_PATH)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.caption(f"Dataset: {DATA_PATH}")

city = st.sidebar.selectbox("City (map focus)", cities_df["city"])
industry = st.sidebar.selectbox("Industry", INDUSTRIES)
employees = st.sidebar.number_input("Employees", 1, 100000, 25)
urgency = st.sidebar.selectbox("Urgency", URGENCY)

proximity_w = st.sidebar.slider("Client proximity", 0, 100, 50) / 100
hiring_w = st.sidebar.slider("Hiring intensity", 0, 100, 50) / 100
cost_w = st.sidebar.slider("Cost sensitivity", 0, 100, 50) / 100
skill_w = st.sidebar.slider("Skill specificity", 0, 100, 50) / 100

# ============================================================
# MAP VIEW STATE
# ============================================================
sel = cities_df[cities_df.city == city].iloc[0]
target_lng, target_lat = clamp_to_uk(float(sel.lng), float(sel.lat))

view_state = pdk.ViewState(
    longitude=target_lng,
    latitude=target_lat,
    zoom=10.5,
    pitch=55,
    bearing=-15
)

# ============================================================
# MODEL SCORING
# ============================================================
# Ensure prediction matrix is numeric and aligned
missing_features = [f for f in feature_list if f not in df.columns]
if missing_features:
    st.warning(
        f"{len(missing_features)} model features were missing in the dataset; "
        "they were added as 0 so scoring can continue."
    )

X = df.reindex(columns=feature_list).copy()
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")
X = X.fillna(X.median(numeric_only=True))
X = X.fillna(0)

base = pipe.predict(X)

approval = minmax_series(df.get("approval_rate", 0))
delay = minmax_series(df.get("median_decision_days", 0))
liq = minmax_series(df.get("job_liquidity_score_1_10", 0))

adj = np.zeros(len(df), dtype=float)

urgency_w = {
    "<3 months": (0.35, 0.40),
    "3-6 months": (0.25, 0.25),
    "6+ months": (0.15, 0.10),
}[urgency]

adj += urgency_w[0] * approval.values - urgency_w[1] * delay.values
adj += (0.10 + hiring_w * 0.10) * liq.values

if "business_density" in df.columns:
    adj += 0.05 * minmax_series(df["business_density"]).values

if "innovation_density" in df.columns:
    adj += 0.03 * skill_w * minmax_series(df["innovation_density"]).values

# Scenario-weighted blend
base_std = float(np.std(base)) + 1e-9
adj_std = float(np.std(adj)) + 1e-9
adj_scaled = adj * (base_std / adj_std)

final_score = 0.50 * base + 0.50 * adj_scaled

df_scored = df[["lad_code", "lad_name"]].copy()
df_scored["score"] = final_score

# Normalize to 0â€“100
df_scored["score"] = 100 * (df_scored["score"] - df_scored["score"].min()) / (
    df_scored["score"].max() - df_scored["score"].min() + 1e-9
)

# ============================================================
# CITY-BASED FILTERING (candidate pool)
# ============================================================
has_centroids = {"lad_lat", "lad_lng"}.issubset(df.columns)

if has_centroids:
    lat = pd.to_numeric(df["lad_lat"], errors="coerce")
    lng = pd.to_numeric(df["lad_lng"], errors="coerce")
    ok = lat.notna() & lng.notna()

    df_local = df_scored.loc[ok].copy()
    df_local["lad_lat"] = lat.loc[ok].values
    df_local["lad_lng"] = lng.loc[ok].values

    df_local["dist_km"] = haversine_km(df_local["lad_lat"].values, df_local["lad_lng"].values, target_lat, target_lng)

    # Nearest-N pool (more stable than a hard radius)
    nearest_n = int(np.clip(220 - proximity_w * 160, 50, 220))
    candidates = df_local.nsmallest(nearest_n, "dist_km").copy()

    top = candidates.sort_values("score", ascending=False).head(15).copy()
else:
    st.warning("No lad_lat/lad_lng found in dataset. City cannot filter table until centroids are added.")
    top = df_scored.sort_values("score", ascending=False).head(15).copy()

# ============================================================
# MAP LAYERS
# ============================================================
# Cloud around selected city; score sampling from top to reflect ranking
rng = np.random.default_rng(7)
cloud = pd.DataFrame({
    "lng": target_lng + rng.normal(0, 0.06, 450),
    "lat": target_lat + rng.normal(0, 0.04, 450),
    "score": rng.choice(top["score"].values, 450, replace=True)
})

hex_layer = pdk.Layer(
    "HexagonLayer",
    cloud,
    get_position=["lng", "lat"],
    radius=1400,
    elevation_scale=30,
    extruded=True,
    pickable=True,
)

deck = pdk.Deck(
    layers=[hex_layer],
    map_style=MAP_STYLE,
    initial_view_state=view_state,
    tooltip={"text": "Heat score: {score}"}
)

# ============================================================
# LAYOUT
# ============================================================
left, right = st.columns([0.45, 0.55])

with right:
    st.pydeck_chart(deck, width="stretch", height=850)

with left:
    st.markdown("### Recommendations")

    if has_centroids and "dist_km" in top.columns:
        show = top[["lad_code", "lad_name", "dist_km", "score"]].copy()
        show = show.rename(columns={"dist_km": "distance_km"})
    else:
        show = top[["lad_code", "lad_name", "score"]].copy()

    st.dataframe(show, width="stretch", height=650)

    with st.expander("Debug"):
        st.write("Dataset:", DATA_PATH)
        st.write("Has centroids:", has_centroids)
        st.write("Selected city:", city, "target:", (target_lng, target_lat))
        if has_centroids and "dist_km" in top.columns:
            st.write("Nearest pool size:", int(np.clip(220 - proximity_w * 160, 50, 220)))
            st.write("Top distance range (km):", float(top["dist_km"].min()), float(top["dist_km"].max()))
