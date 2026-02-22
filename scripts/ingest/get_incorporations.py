import os
import time
import csv
import requests
from datetime import date

API_KEY = os.environ["CH_API_KEY"]
BASE = "https://api.company-information.service.gov.uk/advanced-search/companies"

# CHANGE THESE IF YOU WANT A DIFFERENT WINDOW
INC_FROM = "2025-01-01"
INC_TO   = "2025-12-31"

OUT_FILE = "incorporations.csv"


def fetch_all():
    start_index = 0
    size = 5000   # max allowed
    total_hits = None
    written = 0

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "company_number",
                "company_name",
                "date_of_creation",
                "postal_code"
            ]
        )
        writer.writeheader()

        while True:
            params = {
                "incorporated_from": INC_FROM,
                "incorporated_to": INC_TO,
                "size": size,
                "start_index": start_index,
            }

            r = requests.get(BASE, params=params, auth=(API_KEY, ""), timeout=60)

            if r.status_code == 429:
                print("Rate limited… sleeping")
                time.sleep(2)
                continue

            r.raise_for_status()
            data = r.json()

            if total_hits is None:
                total_hits = int(data.get("hits", 0))
                print("Total incorporations:", total_hits)

            items = data.get("items", [])
            if not items:
                break

            for it in items:
                addr = it.get("registered_office_address") or {}

                writer.writerow({
                    "company_number": it.get("company_number"),
                    "company_name": it.get("company_name"),
                    "date_of_creation": it.get("date_of_creation"),
                    "postal_code": addr.get("postal_code"),
                })

                written += 1

            start_index += len(items)
            print(f"Downloaded {written}/{total_hits}")

            if start_index >= total_hits:
                break

            time.sleep(0.25)

    print("Done. Saved to", OUT_FILE)


if __name__ == "__main__":
    fetch_all()