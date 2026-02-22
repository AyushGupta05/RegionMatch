import os, json, requests

IBEX_HOST = "https://ibex.seractech.co.uk"
JWT_PATH = "ibex_jwt.txt"

DATE_FROM = "2025-01-01"
DATE_TO   = "2025-01-31"

PAGE_SIZE = 1000
MAX_PAGES = 50

def load_jwt():
    base = os.path.dirname(os.path.abspath(__file__))
    return open(os.path.join(base, JWT_PATH), "r", encoding="utf-8").read().strip()

def extract_items(resp):
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        for k in ("items","results","applications","data"):
            if k in resp and isinstance(resp[k], list):
                return resp[k]
    return []

jwt = load_jwt()
headers = {"Authorization": f"Bearer {jwt}", "Content-Type": "application/json"}

all_ids = set()
all_names = {}

for page in range(1, MAX_PAGES + 1):
    payload = {
        "input": {
            "date_from": DATE_FROM,
            "date_to": DATE_TO,
            "page": page,
            "page_size": PAGE_SIZE
            # NOTE: NO council_id here on purpose
        }
    }

    r = requests.post(f"{IBEX_HOST}/applications", headers=headers, json=payload, timeout=60)

    if r.status_code == 400:
        raise SystemExit("400 Bad Request:\n" + r.text)

    r.raise_for_status()
    resp = r.json()
    items = extract_items(resp)

    print(f"Page {page}: {len(items)} items")

    if not items:
        break

    for a in items:
        cid = a.get("council_id")
        cname = a.get("council_name") or a.get("council") or a.get("authority_name")
        if cid is not None:
            all_ids.add(cid)
            if cname:
                all_names[cid] = cname

    if len(items) < PAGE_SIZE:
        break

out = [{"council_id": cid, "council_name": all_names.get(cid)} for cid in sorted(all_ids)]
with open("ibex_council_ids.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"\nFound {len(out)} unique council_ids")
print("Saved: ibex_council_ids.json")