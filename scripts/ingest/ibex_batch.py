import os
import csv
import json
import statistics
from datetime import datetime
import requests

IBEX_HOST = "https://ibex.seractech.co.uk"
JWT_PATH = "ibex_jwt.txt"

# --- Time window for features ---
DATE_FROM = "2025-04-01"
DATE_TO   = "2025-05-01"
DATE_RANGE_TYPE = "validated"

PAGE_SIZE = 1000
MAX_PAGES = 10   # increase if you expect more than 10k apps per council in window

# Put the council_ids you want here:
COUNCIL_IDS = [10, 20, 30]   # <-- replace with your list


def parse_date(x):
    if not x:
        return None
    try:
        return datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except:
        return None


def load_jwt():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return open(os.path.join(base_dir, JWT_PATH), "r", encoding="utf-8").read().strip()


def call_applications(council_id: int, page: int):
    jwt = load_jwt()
    headers = {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}
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


def extract_items(resp):
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        for k in ("items", "results", "applications", "data"):
            if k in resp and isinstance(resp[k], list):
                return resp[k]
    return []


def compute_features(apps):
    approved = 0
    refused = 0
    decision_days = []
    commercial = 0

    keywords = ["restaurant","cafe","café","takeaway","shop","retail","office","warehouse","commercial"]

    for a in apps:
        proposal = (a.get("proposal") or a.get("description") or "")
        proposal_l = str(proposal).lower()
        if any(k in proposal_l for k in keywords):
            commercial += 1

        decision = (a.get("normalised_decision") or a.get("decision") or a.get("status") or "")
        d = str(decision).lower()
        if "approved" in d or "granted" in d:
            approved += 1
        elif "refused" in d:
            refused += 1

        validated = parse_date(a.get("validated_date") or a.get("received_date") or a.get("submission_date"))
        decided   = parse_date(a.get("decision_date"))
        if validated and decided and decided >= validated:
            decision_days.append((decided - validated).days)

    decided_n = approved + refused

    return {
        "apps_total": len(apps),
        "apps_decided": decided_n,
        "approval_rate": (approved / decided_n) if decided_n else None,
        "median_decision_days": statistics.median(decision_days) if decision_days else None,
        "commercial_apps": commercial
    }


def fetch_all_for_council(council_id: int):
    all_apps = []
    for page in range(1, MAX_PAGES + 1):
        resp = call_applications(council_id, page)
        items = extract_items(resp)
        if not items:
            break
        all_apps.extend(items)
        if len(items) < PAGE_SIZE:
            break
    return all_apps


def main():
    out_csv = "ibex_features_by_council.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "council_id",
                "apps_total",
                "apps_decided",
                "approval_rate",
                "median_decision_days",
                "commercial_apps"
            ]
        )
        writer.writeheader()

        for cid in COUNCIL_IDS:
            print("Fetching council:", cid)
            apps = fetch_all_for_council(cid)
            feats = compute_features(apps)
            feats["council_id"] = cid
            writer.writerow(feats)
            print(" ->", feats)

    print("Saved:", out_csv)


if __name__ == "__main__":
    main()