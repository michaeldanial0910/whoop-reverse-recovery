import json
import time
import requests
from pathlib import Path

BASE_URL = "https://api.prod.whoop.com/developer"
ENDPOINTS = {
    "cycles": "/v2/cycle",
    "recovery": "/v2/recovery",
    "sleep": "/v2/activity/sleep",
    "workouts": "/v2/activity/workout",
}

tokens_path = Path(__file__).resolve().parent.parent / ".whoop_tokens.json"
with open(tokens_path) as f:
    tokens = json.load(f)

headers = {"Authorization": f"Bearer {tokens['access_token']}"}


def fetch_all(endpoint_path):
    records = []
    next_token = None
    while True:
        params = {"limit": 25}
        if next_token:
            params["nextToken"] = next_token

        resp = requests.get(f"{BASE_URL}{endpoint_path}", headers=headers, params=params)

        if resp.status_code == 429:
            print("Rate limited, waiting 10s...")
            time.sleep(10)
            continue
        if resp.status_code != 200:
            print(f"Error {resp.status_code} on {endpoint_path}: {resp.text}")
            break

        data = resp.json()
        records.extend(data.get("records", []))
        next_token = data.get("next_token")

        print(f"{endpoint_path}: {len(records)} records so far...")
        time.sleep(0.3)  # stay well under the rate limit

        if not next_token:
            break

    return records


raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

for name, path in ENDPOINTS.items():
    print(f"\nPulling {name}...")
    records = fetch_all(path)
    out_path = raw_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved {len(records)} records to {out_path}")