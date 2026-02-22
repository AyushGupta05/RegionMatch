import joblib
import numpy as np
import pandas as pd

df = pd.read_csv("training_data_v1.csv")
pipe = joblib.load("location_model.joblib")
features = joblib.load("model_features.joblib")

def minmax(s):
    s = s.astype(float)
    s = s.fillna(s.median())
    return (s - s.min()) / (s.max() - s.min() + 1e-9)

# Base score
base = pipe.predict(df[features])
df["base_score"] = base

# Normalized fields for adjustments
if "approval_rate" in df.columns:
    df["approval_norm"] = minmax(df["approval_rate"].fillna(df["approval_rate"].median()))
else:
    df["approval_norm"] = 0

if "median_decision_days" in df.columns:
    df["delay_norm"] = minmax(df["median_decision_days"].fillna(df["median_decision_days"].median()))
else:
    df["delay_norm"] = 0

if "job_liquidity_score_1_10" in df.columns:
    df["liq_norm"] = minmax(df["job_liquidity_score_1_10"].fillna(5))
else:
    df["liq_norm"] = 0

def scenario_score(industry, employees, urgency):
    # urgency weights
    urgency_profiles = {
        "<3 months": {"approval": 0.35, "delay": 0.40},
        "3-6 months": {"approval": 0.25, "delay": 0.25},
        "6+ months": {"approval": 0.15, "delay": 0.10}
    }
    w = urgency_profiles[urgency]

    # employee sensitivity
    liq_scale = 0.05 if employees < 10 else 0.10 if employees < 50 else 0.15

    # industry boosts (use features you already have)
    boosts = {
        "Technology": ["tech_business_density", "core_tech_density"],
        "Creative": ["creative_density"],
        "Innovation": ["innovation_density"],
        "Business Services": ["business_services_density"],
        "Retail/Hospitality": ["commercial_apps"],
        "Industrial/Logistics": ["commercial_apps"]
    }

    adj = np.zeros(len(df))

    # urgency adjustment
    adj += w["approval"] * df["approval_norm"] - w["delay"] * df["delay_norm"]

    # employees adjustment
    adj += liq_scale * df["liq_norm"]

    # industry boost
    for feat in boosts.get(industry, []):
        if feat in df.columns:
            adj += 0.05 * minmax(df[feat].fillna(df[feat].median()))

    return df["base_score"].values + adj

def show_top(industry, employees, urgency, n=10):
    s = scenario_score(industry, employees, urgency)
    tmp = df[["lad_code","lad_name"]].copy()
    tmp["scenario_score"] = s
    tmp = tmp.sort_values("scenario_score", ascending=False).head(n)
    print(f"\n=== TOP {n} for {industry}, employees={employees}, urgency={urgency} ===")
    print(tmp.to_string(index=False))

if __name__ == "__main__":
    show_top("Technology", 8, "<3 months")
    show_top("Technology", 80, "<3 months")
    show_top("Retail/Hospitality", 15, "<3 months")
    show_top("Retail/Hospitality", 15, "6+ months")