"""
Created on Mon May 11 2026
@author: khari
For generating telegram notification
Use this as a template. Telegram token and chat ID need to be set-up by yourself.

Dependency:
- requests
"""

# ─── Import all required dependencies ──────────────────
import requests

TELEGRAM_TOKEN = "<put your token here>"
TELEGRAM_CHAT_ID = "<put your chat ID here>"

def notify(message: str, token: str = TELEGRAM_TOKEN, chat_id: str = TELEGRAM_CHAT_ID):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": message})


