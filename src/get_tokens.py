import os
import requests
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

from whoop_client import TOKEN_URL, TOKENS_PATH, save_tokens

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

CLIENT_ID = os.getenv("WHOOP_CLIENT_ID")
CLIENT_SECRET = os.getenv("WHOOP_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/callback"

raw = input("Paste the code value or the full redirected URL: ").strip()
code = parse_qs(urlparse(raw).query)["code"][0] if "code=" in raw else raw

response = requests.post(
    TOKEN_URL,
    data={
        "grant_type": "authorization_code",
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
    },
    headers={"Content-Type": "application/x-www-form-urlencoded"},
)

if response.status_code == 200:
    # saved with an obtained_at timestamp so whoop_client knows when to refresh;
    # tokens are not printed (terminal history/screenshots would leak them)
    tokens = save_tokens(response.json(), TOKENS_PATH)
    print(f"Success. Scopes: {tokens.get('scope')}. Saved to {TOKENS_PATH}")
    if "refresh_token" not in tokens:
        print("WARNING: no refresh_token returned -- was the 'offline' scope granted?")
else:
    print(f"Error {response.status_code}: {response.text}")
