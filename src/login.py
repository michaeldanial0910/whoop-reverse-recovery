"""One-step WHOOP login: opens the consent page, catches the redirect on localhost, saves tokens.

    python src/login.py

Replaces the two-step auth.py + get_tokens.py flow (kept as a manual fallback): a tiny local server
on the redirect URI receives the authorization code, checks the `state` value, exchanges the code
for tokens and saves them to .whoop_tokens.json (git-ignored). Nothing leaves your machine except
the calls to WHOOP itself.
"""
import os
import secrets
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whoop_client import TOKEN_URL, TOKENS_PATH, save_tokens  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=ROOT / ".env")

REDIRECT_URI = "http://localhost:8000/callback"
AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
SCOPES = "read:recovery read:sleep read:cycles read:workout read:profile read:body_measurement offline"
TIMEOUT_S = 300

PAGE = """<!doctype html><meta charset="utf-8"><title>WHOOP login</title>
<body style="font-family:system-ui;max-width:34rem;margin:5rem auto;color:#16202b">
<h2>{title}</h2><p>{body}</p></body>"""


def credentials():
    cid, secret = os.getenv("WHOOP_CLIENT_ID"), os.getenv("WHOOP_CLIENT_SECRET")
    if not cid or not secret:
        sys.exit("Missing WHOOP_CLIENT_ID / WHOOP_CLIENT_SECRET. Create a .env file in the repo folder "
                 "with both values from your app on the WHOOP developer portal (see README).")
    return cid, secret


def login():
    cid, secret = credentials()
    state = secrets.token_urlsafe(16)
    result = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            if url.path != "/callback":
                self.send_response(404); self.end_headers(); return
            q = parse_qs(url.query)
            if q.get("state", [""])[0] != state:
                result["error"] = "state mismatch (the response did not come from this login attempt)"
            elif "error" in q:
                result["error"] = q["error"][0]
            else:
                result["code"] = q.get("code", [""])[0]
            ok = "code" in result
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.format(
                title="Logged in" if ok else "Login failed",
                body="You can close this tab and go back to the terminal." if ok else result["error"]).encode())
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, *args):          # keep the terminal quiet
            pass

    try:
        server = HTTPServer(("localhost", 8000), Handler)
    except OSError:
        sys.exit("Port 8000 is busy. Close whatever is using it and run this again.")
    server.timeout = TIMEOUT_S
    url = f"{AUTH_URL}?{urlencode({'client_id': cid, 'redirect_uri': REDIRECT_URI, 'response_type': 'code', 'scope': SCOPES, 'state': state})}"
    print("Opening the WHOOP login page in your browser. If it doesn't open, visit:\n" + url)
    webbrowser.open(url)
    timer = threading.Timer(TIMEOUT_S, server.shutdown); timer.daemon = True; timer.start()
    server.serve_forever()
    timer.cancel()
    server.server_close()

    if "code" not in result:
        sys.exit(f"Login failed: {result.get('error', 'no response within 5 minutes')}")
    resp = requests.post(TOKEN_URL, timeout=30, data={
        "grant_type": "authorization_code", "code": result["code"], "client_id": cid,
        "client_secret": secret, "redirect_uri": REDIRECT_URI})
    if resp.status_code != 200:
        sys.exit(f"Token exchange failed ({resp.status_code}): {resp.text[:300]}")
    tokens = save_tokens(resp.json(), TOKENS_PATH)          # tokens are never printed
    if "refresh_token" not in tokens:
        print("WARNING: no refresh token returned; was the 'offline' scope granted?")
    print(f"Logged in. Tokens saved to {TOKENS_PATH.name} (git-ignored).")


if __name__ == "__main__":
    login()
