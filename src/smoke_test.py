import json
import requests
from pathlib import Path

tokens_path = Path(__file__).resolve().parent.parent / ".whoop_tokens.json"
with open(tokens_path) as f:
    tokens = json.load(f)

access_token = tokens["access_token"]

response = requests.get(
    "https://api.prod.whoop.com/developer/v2/user/profile/basic",
    headers={"Authorization": f"Bearer {access_token}"},
)

print(f"Status: {response.status_code}")
print(response.json())