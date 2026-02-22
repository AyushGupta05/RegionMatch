import os
import json
import statistics
from datetime import datetime
import requests

IBEX_HOST = "https://ibex.seractech.co.uk"
JWT_PATH = "ibex_jwt.txt"

COUNCIL_ID = 10
DATE_FROM = "2025-04-01"
DATE_TO   = "2025-05-01"
DATE_RANGE_TYPE = "validated"
PAGE_SIZE = 1000
MAX_PAGES = 5

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

def call_applications(page: int):
    jwt = load_jwt()
    headers = {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}

    payload = {
        "input": {
            "date_range_type": DATE_RANGE_TYPE,
            "date_to": DATE_TO,
            "date_from": DATE_FROM,
            "council_id": [COUNCIL_ID],
            "page": page,
            "page_size": PAGE_SIZE
        }
    }

    r = requests.post(f"{IBEX_HOST}/applications", headers=headers, json=payload, timeout=60)

    print("HTTP:", r.status_code)
    if r.status_code != 200:
        print("Response text:", r.text[:500])
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

def main():
    print("RUNNING FROM:", os.getcwd())
    print("SCRIPT DIR  :", BASE_DIR)

    # guaranteed test write
    test_path = os.path.join(BASE_DIR, "TEST_WRITE.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        f.write("hello")
    print("WROTE:", test_path)

    all_apps = []
    for page in range(1, MAX_PAGES + 1):
        resp = call_applications(page)
        items = extract_items(resp)
        print("PAGE", page, "ITEMS", len(items))
        if not items:
            break
        all_apps.extend(items)
        if len(items) < PAGE_SIZE:
            break

    feats = compute_features(all_apps)
    print("FEATURES:", json.dumps(feats, indent=2))

    raw_path = os.path.join(BASE_DIR, "ibex_apps_raw.json")
    feat_path = os.path.join(BASE_DIR, "ibex_features.json")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_apps, f)

    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feats, f, indent=2)

    print("WROTE:", raw_path)
    print("WROTE:", feat_path)

if __name__ == "__main__":
    main()