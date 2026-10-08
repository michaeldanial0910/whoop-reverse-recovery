"""Daily sync: pull new WHOOP data, merge it into data/raw/*.json, rebuild the tables.

  python src/sync.py            incremental: re-pulls the last `overlap_days` and merges
  python src/sync.py --full     full history (what pull_history.py used to do)
  python src/sync.py --no-rebuild   only update data/raw, skip normalize/features

Why an overlap instead of "only what's new": WHOOP edits recent records after
the fact (the in-progress cycle gets its end time and final strain, scores get
re-processed, a sleep can be re-classified). Re-pulling a few days and letting
the newer copy of each record win keeps those edits.

All-or-nothing: every endpoint is fetched into memory first; raw files are only
written once all four succeeded, each via an atomic rename. A failed run leaves
yesterday's data untouched instead of a mix of old and half-new files.

Known limit: a record WHOOP deletes stays in our copy until the next --full run.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from whoop_client import ENDPOINTS, WhoopClient, write_json_atomic

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

CONFIG = {
    "overlap_days": 7,
}


def record_key(rec):
    # cycles, sleeps and workouts have their own id; recovery records are keyed by cycle
    return str(rec["id"]) if "id" in rec else f"cycle:{rec['cycle_id']}"


def merge_records(old, new):
    """Union by record key; the freshly pulled copy wins. Sorted newest-first like the API."""
    merged = {record_key(r): r for r in old}
    merged.update({record_key(r): r for r in new})
    return sorted(merged.values(), key=lambda r: r.get("start") or r.get("created_at") or "", reverse=True)


def load_raw(name):
    path = RAW_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else []


def incremental_start(existing_cycles):
    """ISO start for the incremental pull, or None (full pull) if there is no history yet."""
    starts = [c["start"] for c in existing_cycles if c.get("start")]
    if not starts:
        return None
    latest = datetime.fromisoformat(max(starts).replace("Z", "+00:00"))
    since = latest - timedelta(days=CONFIG["overlap_days"])
    return since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def sync(full=False, client=None):
    client = client or WhoopClient()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = {name: ([] if full else load_raw(name)) for name in ENDPOINTS}
    start = None if full else incremental_start(existing["cycles"])
    print(f"Sync mode: {'full history' if start is None else f'incremental from {start}'}")

    pulled = {}
    for name, path in ENDPOINTS.items():          # fetch everything before writing anything
        pulled[name] = client.fetch_all(path, start=start)
        print(f"  {name}: {len(pulled[name])} records pulled")

    summary = {}
    for name in ENDPOINTS:
        merged = merge_records(existing[name], pulled[name])
        summary[name] = len(merged) - len(existing[name])
        write_json_atomic(RAW_DIR / f"{name}.json", merged)
    print("New records:", summary)
    return summary


def rebuild():
    for script in ["normalize_data.py", "features.py"]:
        subprocess.run([sys.executable, str(ROOT / "src" / script)], check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="re-pull the full history")
    parser.add_argument("--no-rebuild", action="store_true", help="skip normalize_data/features")
    args = parser.parse_args()
    print(f"[{datetime.now().isoformat(timespec='seconds')}] WHOOP sync")
    sync(full=args.full)
    if not args.no_rebuild:
        rebuild()


if __name__ == "__main__":
    main()
