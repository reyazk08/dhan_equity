import os
from dhanhq import dhanhq
from dotenv import load_dotenv

# 🔄 Load variables from .env
load_dotenv()

# 🔐 Read credentials
client_id = os.getenv("CLIENT_ID")
access_token = os.getenv("ACCESS_TOKEN")

# ✅ Initialize Dhan API client
dhan = dhanhq(client_id, access_token)

# 🔍 Test connection: Get fund limits
try:
    fund_limits = dhan.get_fund_limits()
    print("✅ Connected to Dhan successfully!")
    print("💰 Fund Limits:", fund_limits)
except Exception as e:
    print("❌ Error connecting to Dhan:", e)