"""Pull the full WHOOP history into data/raw/*.json.

Kept for the M0 workflow; the logic now lives in sync.py (token refresh,
bounded retries, all-or-nothing atomic writes). Equivalent to:
    python src/sync.py --full --no-rebuild
"""
from sync import sync

if __name__ == "__main__":
    sync(full=True)
