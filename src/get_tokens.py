import os
import json
import requests
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

CLIENT_ID = os.getenv("WHOOP_CLIENT_ID")
CLIENT_SECRET = os.getenv("WHOOP_CLIENT_SECRET")
REDIRECT_URI = "http://localhost:8000/callback"
TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"

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
    tokens = response.json()
    print("Success:")
    print(json.dumps(tokens, indent=2))

    tokens_path = Path(__file__).resolve().parent.parent / ".whoop_tokens.json"
    with open(tokens_path, "w") as f:
        json.dump(tokens, f, indent=2)
    print(f"Saved to {tokens_path}")
else:
    print(f"Error {response.status_code}: {response.text}")