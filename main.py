# main.py
"""
Minimal text-only GPT Telegram bot, ready for Render deployment.

✓ Supports POST, GET & HEAD on /webhook/{token}
✓ Adds a root “/” health-check (Render pings this)
✓ Uses async OpenAI client (>= 1.14)

Build command on Render:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000

Environment variables to add on Render:
  TELEGRAM_BOT_TOKEN   # your bot token
  OPENAI_API_KEY       # your OpenAI secret key
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import httpx
import openai
from fastapi import FastAPI, HTTPException, Request

# ────────────────────────────────────────────────────────────────────────────
# Environment
# ────────────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError(
        "Set TELEGRAM_BOT_TOKEN and OPENAI_API_KEY as environment variables."
    )

openai.api_key = OPENAI_API_KEY
logging.basicConfig(level=logging.INFO)

app = FastAPI()


# ────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────
async def send_telegram_message(chat_id: int, text: str) -> None:
    """Send a plaintext message back to the user via Telegram."""
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )


async def generate_reply(prompt: str) -> str:
    """Generate a conversational reply via OpenAI."""
    completion = await openai.ChatCompletion.acreate(
        model="gpt-4o-mini",   # adjust to whatever model name your key supports
        messages=[
            {"role": "system", "content": "You are a friendly English tutor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return completion.choices[0].message.content.strip()


# ────────────────────────────────────────────────────────────────────────────
# Webhook endpoint
# ────────────────────────────────────────────────────────────────────────────
@app.api_route("/webhook/{token}", methods=["POST", "GET", "HEAD"])
async def telegram_webhook(token: str, request: Request):
    """
    • Telegram calls GET/HEAD right after setWebhook → return 200 OK
    • Telegram sends updates as POST JSON payloads
    """
    if token != TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token in path")

    # Handshake request
    if request.method in ("GET", "HEAD"):
        return {"ok": True}

    # Incoming update
    update: dict = await request.json()

    message = update.get("message", {})
    text: str | None = message.get("text")
    chat_id: int | None = message.get("chat", {}).get("id")

    if chat_id is None or text is None:
        logging.info("Ignoring non-text update: %s", update)
        return {"ok": True}

    logging.info("Received message from %s: %s", chat_id, text)

    try:
        reply_text = await generate_reply(text)
    except Exception as exc:  # noqa: BLE001
        logging.exception("OpenAI request failed: %s", exc)
        reply_text = (
            "Sorry, I’m having trouble generating a reply right now. "
            "Please try again later."
        )

    await send_telegram_message(chat_id, reply_text)
    logging.info("Sent reply to %s", chat_id)

    return {"ok": True}


# ────────────────────────────────────────────────────────────────────────────
# Health-check root
# ────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root() -> dict[str, Literal["pong"]]:
    """Render health-check."""
    return {"status": "pong"}
