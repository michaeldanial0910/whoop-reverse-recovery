from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv("WHOOP_CLIENT_ID"))
