"""Minimal WHOOP API v2 client: token storage, token refresh, paginated fetch.

Fixes the three M0 fragilities:
  - token refresh: access tokens last ~1 h; each refresh returns a NEW refresh
    token too, so both are saved every time (the old refresh token stops working)
  - bounded retries: 429/5xx are retried with backoff a fixed number of times,
    then the run fails loudly instead of looping forever
  - no silent partial data: any non-retryable error raises; callers write
    nothing unless every endpoint succeeded (see sync.py)

Credentials come from .env (WHOOP_CLIENT_ID, WHOOP_CLIENT_SECRET); tokens live
in .whoop_tokens.json (git-ignored). Run auth.py + get_tokens.py once first.
"""
import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
TOKENS_PATH = ROOT / ".whoop_tokens.json"
load_dotenv(dotenv_path=ROOT / ".env")

BASE_URL = "https://api.prod.whoop.com/developer"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
ENDPOINTS = {
    "cycles": "/v2/cycle",
    "recovery": "/v2/recovery",
    "sleep": "/v2/activity/sleep",
    "workouts": "/v2/activity/workout",
}

CONFIG = {
    "page_limit": 25,            # API maximum per page
    "page_pause_s": 0.3,         # stay well under the 100 requests/minute limit
    "max_retries": 5,            # per request, for 429 and 5xx
    "backoff_base_s": 2.0,       # 2, 4, 8, 16, 32 s unless the server sends Retry-After
    "refresh_margin_s": 300,     # refresh if the access token expires within 5 min
    "timeout_s": 30,
}


class WhoopAPIError(RuntimeError):
    pass


def write_json_atomic(path, obj):
    """Write to a temp file, then rename over the target.

    os.replace is atomic on the same filesystem: a crash mid-write leaves the
    old file intact instead of a truncated one.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def save_tokens(token_response, path=TOKENS_PATH):
    tokens = dict(token_response)
    tokens["obtained_at"] = time.time()  # expires_in is relative; store when we got it
    write_json_atomic(path, tokens)
    return tokens


class WhoopClient:
    def __init__(self, tokens_path=TOKENS_PATH, session=None, sleep=time.sleep):
        self.tokens_path = Path(tokens_path)
        if not self.tokens_path.exists():
            raise WhoopAPIError(f"No tokens at {self.tokens_path}. Run src/auth.py then src/get_tokens.py first.")
        self.tokens = json.loads(self.tokens_path.read_text())
        self.session = session or requests.Session()
        self._sleep = sleep  # injectable so tests don't actually wait

    # --- tokens -----------------------------------------------------------
    def _token_expiring(self):
        obtained = self.tokens.get("obtained_at")
        if obtained is None:  # tokens saved by the M0 script: age unknown, assume stale
            return True
        expires_at = obtained + self.tokens.get("expires_in", 3600)
        return time.time() > expires_at - CONFIG["refresh_margin_s"]

    def refresh(self):
        client_id, client_secret = os.getenv("WHOOP_CLIENT_ID"), os.getenv("WHOOP_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise WhoopAPIError("WHOOP_CLIENT_ID / WHOOP_CLIENT_SECRET missing from .env")
        resp = self.session.post(TOKEN_URL, timeout=CONFIG["timeout_s"], data={
            "grant_type": "refresh_token",
            "refresh_token": self.tokens["refresh_token"],
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "offline",
        })
        if resp.status_code != 200:
            raise WhoopAPIError(
                f"Token refresh failed ({resp.status_code}): {resp.text[:300]}\n"
                "If the refresh token is no longer valid, re-run src/auth.py and src/get_tokens.py.")
        # save BOTH tokens immediately: the old refresh token is now dead
        self.tokens = save_tokens(resp.json(), self.tokens_path)

    # --- requests -----------------------------------------------------------
    def get(self, path, params=None):
        if self._token_expiring():
            self.refresh()
        refreshed_on_401 = False
        for attempt in range(CONFIG["max_retries"] + 1):
            resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=CONFIG["timeout_s"],
                                    headers={"Authorization": f"Bearer {self.tokens['access_token']}"})
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 401 and not refreshed_on_401:
                self.refresh()            # token revoked/expired early: one refresh, then retry
                refreshed_on_401 = True
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == CONFIG["max_retries"]:
                    break
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() \
                    else CONFIG["backoff_base_s"] * 2 ** attempt
                print(f"  {resp.status_code} on {path}, retrying in {wait:.0f}s "
                      f"({attempt + 1}/{CONFIG['max_retries']})")
                self._sleep(wait)
                continue
            raise WhoopAPIError(f"{resp.status_code} on {path}: {resp.text[:300]}")
        raise WhoopAPIError(f"Gave up on {path} after {CONFIG['max_retries']} retries "
                            f"(last status {resp.status_code})")

    def fetch_all(self, path, start=None):
        """All records from a paginated collection endpoint, optionally from `start` (ISO-8601 UTC)."""
        records, next_token = [], None
        while True:
            params = {"limit": CONFIG["page_limit"]}
            if start:
                params["start"] = start
            if next_token:
                params["nextToken"] = next_token
            data = self.get(path, params)
            records.extend(data.get("records", []))
            next_token = data.get("next_token")
            if not next_token:
                return records
            self._sleep(CONFIG["page_pause_s"])
