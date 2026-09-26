import os
import secrets
import webbrowser
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("WHOOP_CLIENT_ID")
REDIRECT_URI = "http://localhost:8000/callback"
AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
SCOPES = "read:recovery read:sleep read:cycles read:workout read:profile read:body_measurement offline"

state = secrets.token_urlsafe(16)

params = {
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "response_type": "code",
    "scope": SCOPES,
    "state": state,
}

from urllib.parse import urlencode
full_url = f"{AUTH_URL}?{urlencode(params)}"

print(f"State (save this, we'll check it): {state}")
print(f"Opening: {full_url}")
webbrowser.open(full_url)