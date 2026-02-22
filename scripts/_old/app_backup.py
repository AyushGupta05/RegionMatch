import os
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
    page_icon="🧭",
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
# CITY DATA
# ============================================================
UK_CITIES = [
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
    ("Edinburgh", -3.1883, 55.9533),
    ("Glasgow", -4.2518, 55.8642),
    ("Cardiff", -3.1791, 51.4816),
    ("Belfast", -5.9301, 54.5973),
    ("Southampton", -1.4043, 50.9097),
    ("Brighton", -0.1364, 50.8225),
    ("Cambridge", 0.1218, 52.2053),
    ("Oxford", -1.2577, 51.7520),
    ("York", -1.0815, 53.9590),
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

UK_CENTER = (-2.5, 54.5)
UK_ZOOM_DEFAULT = 5.4

# ============================================================
# HELPERS
# ============================================================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def clamp_to_uk(lng, lat):
    return clamp(lng, -8.8, 2.3), clamp(lat, 49.8, 60.9)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def minmax_series(s):
    s = pd.to_numeric(s, errors="coerce")
    if s.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())

# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("location_model.joblib")
    features = joblib.load("model_features.joblib")
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv("training_data_v1.csv")

pipe, feature_list = load_model()
df = load_data()

# ============================================================
# SIDEBAR
# ============================================================
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
target_lng, target_lat = clamp_to_uk(sel.lng, sel.lat)

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
base = pipe.predict(df[feature_list])

approval = minmax_series(df.get("approval_rate", 0))
delay = minmax_series(df.get("median_decision_days", 0))
liq = minmax_series(df.get("job_liquidity_score_1_10", 0))

adj = np.zeros(len(df))

urgency_w = {
    "<3 months": (0.35, 0.40),
    "3-6 months": (0.25, 0.25),
    "6+ months": (0.15, 0.10),
}[urgency]

adj += urgency_w[0]*approval.values - urgency_w[1]*delay.values
adj += (0.1 + hiring_w*0.1) * liq.values

if "business_density" in df.columns:
    adj += 0.05 * minmax_series(df["business_density"]).values

if "innovation_density" in df.columns:
    adj += 0.03 * skill_w * minmax_series(df["innovation_density"]).values

# -------- Scenario-weighted blend (KEY FIX) --------
base_std = np.std(base) + 1e-9
adj_std = np.std(adj) + 1e-9
adj_scaled = adj * (base_std / adj_std)

final_score = 0.5 * base + 0.5 * adj_scaled

df_scored = df[["lad_code", "lad_name"]].copy()
df_scored["score"] = final_score

# Normalize
df_scored["score"] = 100 * (df_scored["score"] - df_scored["score"].min()) / (
    df_scored["score"].max() - df_scored["score"].min() + 1e-9
)

# ============================================================
# CITY-BASED FILTERING (FIXED)
# ============================================================
if {"lad_lat", "lad_lng"}.issubset(df.columns):
    df_scored["lad_lat"] = df["lad_lat"]
    df_scored["lad_lng"] = df["lad_lng"]
    df_scored["dist_km"] = haversine_km(
        df_scored["lad_lat"], df_scored["lad_lng"],
        target_lat, target_lng
    )

    radius = int(220 - 140*proximity_w)
    local = df_scored.nsmallest(120, "dist_km")
else:
    local = df_scored

top = local.sort_values("score", ascending=False).head(15)

# ============================================================
# MAP LAYERS
# ============================================================
cloud = pd.DataFrame({
    "lng": target_lng + np.random.normal(0, 0.06, 400),
    "lat": target_lat + np.random.normal(0, 0.04, 400),
    "score": np.random.choice(top.score, 400)
})

hex_layer = pdk.Layer(
    "HexagonLayer",
    cloud,
    get_position=["lng", "lat"],
    radius=1400,
    elevation_scale=30,
    extruded=True,
)

deck = pdk.Deck(
    layers=[hex_layer],
    map_style=MAP_STYLE,
    initial_view_state=view_state
)

# ============================================================
# LAYOUT
# ============================================================
left, right = st.columns([0.45, 0.55])

with right:
    st.pydeck_chart(deck, width="stretch", height=850)

with left:
    st.markdown("### Recommendations")
    st.dataframe(top[["lad_code", "lad_name", "score"]], width="stretch", height=600)