# telegram_gpt_bot/main.py
"""Minimal text‑only GPT Telegram bot for Render deployment.

Prerequisites (Render → Build Command):
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000

Environment variables required in Render:
  TELEGRAM_BOT_TOKEN   # Your bot token, e.g. 123456:ABC‑DEF…
  OPENAI_API_KEY       # Your OpenAI secret key

After deploy, set the webhook:
  https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook?url=https://<render-app>.onrender.com/webhook/<TELEGRAM_BOT_TOKEN>
"""

import os

import httpx
import openai
from fastapi import FastAPI, HTTPException, Request

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError("Environment variables TELEGRAM_BOT_TOKEN and OPENAI_API_KEY must be set.")

openai.api_key = OPENAI_API_KEY

app = FastAPI()

async def send_telegram_message(chat_id: int, text: str) -> None:
    """Send a plain‑text message back to the user via Telegram."""
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )

async def generate_reply(prompt: str) -> str:
    """Generate a conversational reply using GPT‑4‑mini."""
    completion = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",  # adjust if you’re on a different tier
        messages=[
            {"role": "system", "content": "You are a friendly English tutor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content.strip()

@app.post("/webhook/{token}")
async def telegram_webhook(token: str, request: Request):
    """Receive Telegram updates, generate a GPT reply, and send it back."""
    if token != TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token in path")

    update = await request.json()

    message = update.get("message", {})
    text = message.get("text")
    chat = message.get("chat", {})
    chat_id = chat.get("id")

    # Ignore any non‑text messages
    if chat_id is None or text is None:
        return {"ok": True}

    reply_text = await generate_reply(text)
    await send_telegram_message(chat_id, reply_text)

    return {"ok": True}

# ---- requirements.txt ----
# fastapi
# uvicorn[standard]
# httpx
# openai>=1.14  # openai python sdk >= 1.14 has async support
# -------------------------
