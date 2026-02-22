import os
import json
import csv
import statistics
from datetime import datetime
import requests

IBEX_HOST = "https://ibex.seractech.co.uk"
JWT_PATH = "ibex_jwt.txt"
COUNCILS_JSON = "ibex_council_ids.json"

# ====== FEATURE WINDOW ======
DATE_FROM = "2024-01-01"
DATE_TO   = "2024-12-31"
DATE_RANGE_TYPE = "validated"
# ============================

PAGE_SIZE = 1000
MAX_PAGES = 50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_date(x):
    if not x:
        return None
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except Exception:
        return None

def load_jwt():
    return open(os.path.join(BASE_DIR, JWT_PATH), "r", encoding="utf-8").read().strip()

def extract_items(resp):
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        for k in ("items", "results", "applications", "data"):
            if k in resp and isinstance(resp[k], list):
                return resp[k]
    return []

def call_applications(council_id, page):
    jwt = load_jwt()
    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": {
            "date_range_type": DATE_RANGE_TYPE,
            "date_from": DATE_FROM,
            "date_to": DATE_TO,
            "council_id": [council_id],
            "page": page,
            "page_size": PAGE_SIZE
        }
    }

    r = requests.post(f"{IBEX_HOST}/applications", headers=headers, json=payload, timeout=60)

    if r.status_code in (401, 403):
        raise SystemExit(f"Auth error {r.status_code}: {r.text}")
    if r.status_code == 400:
        raise SystemExit(f"Bad request 400: {r.text}")

    r.raise_for_status()
    return r.json()

def fetch_all_apps(council_id):
    apps = []
    for page in range(1, MAX_PAGES + 1):
        resp = call_applications(council_id, page)
        items = extract_items(resp)
        if not items:
            break
        apps.extend(items)
        if len(items) < PAGE_SIZE:
            break
    return apps

def compute_features(apps):
    approved = 0
    refused = 0
    decision_days = []
    commercial = 0

    keywords = [
        "restaurant", "cafe", "café", "takeaway",
        "shop", "retail", "office", "warehouse", "commercial"
    ]

    for a in apps:
        proposal = (a.get("proposal") or a.get("description") or "")
        p = str(proposal).lower()
        if any(k in p for k in keywords):
            commercial += 1

        decision = (a.get("normalised_decision") or a.get("raw_decision") or a.get("decision") or a.get("status") or "")
        d = str(decision).lower()

        if "approved" in d or "granted" in d:
            approved += 1
        elif "refused" in d:
            refused += 1

        # IMPORTANT: these are the actual fields in your raw data
        start_dt = parse_date(a.get("application_date"))
        end_dt   = parse_date(a.get("decided_date"))

        if start_dt and end_dt and end_dt >= start_dt:
            decision_days.append((end_dt - start_dt).days)

    decided_n = approved + refused

    return {
        "apps_total": len(apps),
        "apps_decided": decided_n,
        "approval_rate": (approved / decided_n) if decided_n else None,
        "median_decision_days": statistics.median(decision_days) if decision_days else None,
        "commercial_apps": commercial
    }

def main():
    councils_path = os.path.join(BASE_DIR, COUNCILS_JSON)
    councils = json.load(open(councils_path, "r", encoding="utf-8"))

    out_csv = os.path.join(BASE_DIR, "ibex_features_by_council.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "council_id", "council_name",
                "apps_total", "apps_decided", "approval_rate",
                "median_decision_days", "commercial_apps"
            ]
        )
        writer.writeheader()

        for i, c in enumerate(councils, start=1):
            cid = c.get("council_id")
            cname = c.get("council_name")
            print(f"[{i}/{len(councils)}] Council {cid} - {cname}")

            apps = fetch_all_apps(cid)
            feats = compute_features(apps)

            row = {
                "council_id": cid,
                "council_name": cname,
                **feats
            }
            writer.writerow(row)

    print("Saved:", out_csv)

if __name__ == "__main__":
    main()