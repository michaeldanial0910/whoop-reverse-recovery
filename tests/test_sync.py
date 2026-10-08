"""Offline tests for whoop_client.py and sync.py (no network, no real tokens).

    python -m unittest discover tests
"""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
os.environ.setdefault("WHOOP_CLIENT_ID", "test-id")
os.environ.setdefault("WHOOP_CLIENT_SECRET", "test-secret")

import sync  # noqa: E402
import whoop_client  # noqa: E402
from whoop_client import WhoopAPIError, WhoopClient  # noqa: E402


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


class FakeSession:
    """Returns queued responses in order and records every call."""
    def __init__(self, gets=(), posts=()):
        self.gets, self.posts, self.calls = list(gets), list(posts), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(("GET", url, dict(params or {}), headers))
        return self.gets.pop(0)

    def post(self, url, data=None, timeout=None):
        self.calls.append(("POST", url, data, None))
        return self.posts.pop(0)


def fresh_tokens(path, **overrides):
    tokens = {"access_token": "A1", "refresh_token": "R1", "expires_in": 3600, "obtained_at": time.time()}
    tokens.update(overrides)
    path.write_text(json.dumps(tokens))


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tokens_path = Path(self.tmp.name) / "tokens.json"
        self.waits = []

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, session):
        return WhoopClient(self.tokens_path, session=session, sleep=self.waits.append)

    def test_expired_token_is_refreshed_and_both_tokens_saved(self):
        fresh_tokens(self.tokens_path, obtained_at=time.time() - 7200)
        s = FakeSession(gets=[FakeResponse(200, {"records": []})],
                        posts=[FakeResponse(200, {"access_token": "A2", "refresh_token": "R2", "expires_in": 3600})])
        self.client(s).get("/v2/cycle")
        saved = json.loads(self.tokens_path.read_text())
        self.assertEqual((saved["access_token"], saved["refresh_token"]), ("A2", "R2"))
        self.assertIn("obtained_at", saved)
        self.assertEqual(s.calls[1][3]["Authorization"], "Bearer A2")

    def test_legacy_tokens_without_timestamp_are_refreshed(self):
        fresh_tokens(self.tokens_path)
        tokens = json.loads(self.tokens_path.read_text()); del tokens["obtained_at"]
        self.tokens_path.write_text(json.dumps(tokens))
        s = FakeSession(gets=[FakeResponse(200, {})],
                        posts=[FakeResponse(200, {"access_token": "A2", "refresh_token": "R2"})])
        self.client(s).get("/x")
        self.assertEqual(s.calls[0][0], "POST")

    def test_429_retries_are_bounded(self):
        fresh_tokens(self.tokens_path)
        n = whoop_client.CONFIG["max_retries"] + 1
        s = FakeSession(gets=[FakeResponse(429) for _ in range(n)])
        with self.assertRaises(WhoopAPIError):
            self.client(s).get("/v2/cycle")
        self.assertEqual(len(s.calls), n)
        self.assertEqual(self.waits, [2.0, 4.0, 8.0, 16.0, 32.0])

    def test_retry_after_header_is_respected(self):
        fresh_tokens(self.tokens_path)
        s = FakeSession(gets=[FakeResponse(429, headers={"Retry-After": "7"}), FakeResponse(200, {"ok": 1})])
        self.assertEqual(self.client(s).get("/x"), {"ok": 1})
        self.assertEqual(self.waits, [7.0])

    def test_401_triggers_one_refresh_then_fails(self):
        fresh_tokens(self.tokens_path)
        s = FakeSession(gets=[FakeResponse(401), FakeResponse(401)],
                        posts=[FakeResponse(200, {"access_token": "A2", "refresh_token": "R2"})])
        with self.assertRaises(WhoopAPIError):
            self.client(s).get("/x")
        self.assertEqual(sum(c[0] == "POST" for c in s.calls), 1)

    def test_other_errors_raise_instead_of_returning_partial_data(self):
        fresh_tokens(self.tokens_path)
        s = FakeSession(gets=[FakeResponse(200, {"records": [{"id": 1}], "next_token": "p2"}),
                              FakeResponse(400, {"error": "bad"})])
        with self.assertRaises(WhoopAPIError):
            self.client(s).fetch_all("/v2/cycle")

    def test_pagination_and_start_param(self):
        fresh_tokens(self.tokens_path)
        s = FakeSession(gets=[FakeResponse(200, {"records": [{"id": 1}], "next_token": "p2"}),
                              FakeResponse(200, {"records": [{"id": 2}]})])
        recs = self.client(s).fetch_all("/v2/cycle", start="2026-01-01T00:00:00.000Z")
        self.assertEqual([r["id"] for r in recs], [1, 2])
        self.assertEqual(s.calls[1][2], {"limit": 25, "start": "2026-01-01T00:00:00.000Z", "nextToken": "p2"})


class FakeClient:
    def __init__(self, data, fail_on=None):
        self.data, self.fail_on, self.starts = data, fail_on, []

    def fetch_all(self, path, start=None):
        self.starts.append(start)
        if path == self.fail_on:
            raise WhoopAPIError("boom")
        return self.data.get(path, [])


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.orig_raw = sync.RAW_DIR
        sync.RAW_DIR = Path(self.tmp.name)
        self.old = {
            "cycles": [{"id": 1, "start": "2026-09-20T03:00:00.000Z", "end": None, "score": {"strain": 3}}],
            "recovery": [{"cycle_id": 1, "sleep_id": "s1", "score": {"recovery_score": 50}}],
            "sleep": [{"id": "s1", "start": "2026-09-20T03:00:00.000Z"}],
            "workouts": [],
        }
        for name, recs in self.old.items():
            (sync.RAW_DIR / f"{name}.json").write_text(json.dumps(recs))

    def tearDown(self):
        sync.RAW_DIR = self.orig_raw
        self.tmp.cleanup()

    def read(self, name):
        return json.loads((sync.RAW_DIR / f"{name}.json").read_text())

    def test_incremental_merge_updates_and_appends(self):
        new = {
            "/v2/cycle": [{"id": 1, "start": "2026-09-20T03:00:00.000Z", "end": "2026-09-21T03:00:00.000Z",
                           "score": {"strain": 12}},
                          {"id": 2, "start": "2026-09-21T03:00:00.000Z", "end": None}],
            "/v2/recovery": [{"cycle_id": 2, "sleep_id": "s2", "score": {"recovery_score": 70}}],
        }
        client = FakeClient(new)
        summary = sync.sync(client=client)
        cycles = self.read("cycles")
        self.assertEqual([c["id"] for c in cycles], [2, 1])                     # newest first
        self.assertEqual(next(c for c in cycles if c["id"] == 1)["score"]["strain"], 12)  # edit kept
        self.assertEqual(len(self.read("recovery")), 2)
        self.assertEqual(summary["cycles"], 1)
        self.assertEqual(client.starts[0], "2026-09-13T03:00:00.000Z")         # 7-day overlap

    def test_failed_endpoint_writes_nothing(self):
        before = {n: self.read(n) for n in self.old}
        client = FakeClient({"/v2/cycle": [{"id": 9, "start": "2026-09-25T00:00:00.000Z"}]},
                            fail_on="/v2/activity/sleep")
        with self.assertRaises(WhoopAPIError):
            sync.sync(client=client)
        self.assertEqual({n: self.read(n) for n in self.old}, before)
        self.assertEqual(list(sync.RAW_DIR.glob("*.tmp")), [])

    def test_no_history_means_full_pull(self):
        for p in sync.RAW_DIR.glob("*.json"):
            p.unlink()
        client = FakeClient({})
        sync.sync(client=client)
        self.assertEqual(set(client.starts), {None})


if __name__ == "__main__":
    unittest.main()
