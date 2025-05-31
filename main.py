# main.py
"""
Minimal text‑only GPT Telegram bot (FastAPI) using the **new OpenAI Python ≥ 1.14 SDK**.

✓ Supports GET / HEAD / POST on /webhook/{token}
✓ Root “/” health‑check for Render
✓ Async OpenAI client (AsyncOpenAI)

Build command (Render):
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000

Env vars (set in Render → Environment):
  TELEGRAM_BOT_TOKEN   # your bot token
  OPENAI_API_KEY       # your OpenAI secret key
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI

# ────────────────────────────────────────────────────────────────────────────
# Environment
# ────────────────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_BOT_TOKEN or not OPENAI_API_KEY:
    raise RuntimeError(
        "Environment variables TELEGRAM_BOT_TOKEN and OPENAI_API_KEY must be set."
    )

client = AsyncOpenAI(api_key=OPENAI_API_KEY)
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
async def send_telegram_message(chat_id: int, text: str) -> None:
    """Send plain‑text back to the user via Telegram."""
    async with httpx.AsyncClient(timeout=15) as hc:
        await hc.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )


async def generate_reply(prompt: str) -> str:
    """Generate a conversational reply with GPT‑4‑mini (new SDK syntax)."""
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",  # adjust to whatever model your key supports
        messages=[
            {"role": "system", "content": "You are a friendly English tutor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# ────────────────────────────────────────────────────────────────────────────
# Webhook endpoint
# ────────────────────────────────────────────────────────────────────────────
@app.api_route("/webhook/{token}", methods=["POST", "GET", "HEAD"])
async def telegram_webhook(token: str, request: Request):
    """Handle Telegram webhook handshake & message updates."""

    if token != TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token in path")

    # 1. Handshake requests (Telegram calls GET/HEAD after setWebhook)
    if request.method in ("GET", "HEAD"):
        return {"ok": True}

    # 2. Normal update (POST)
    update: dict = await request.json()

    message = update.get("message", {})
    text: str | None = message.get("text")
    chat_id: int | None = message.get("chat", {}).get("id")

    if chat_id is None or text is None:
        logging.info("Ignoring non‑text update: %s", update)
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
# Health‑check
# ────────────────────────────────────────────────────────────────────────────
@app.get("/")
async def root() -> dict[str, Literal["pong"]]:
    """Simple health‑check endpoint for Render."""
    return {"status": "pong"}

# ────────────────────────────────────────────────────────────────────────────
# requirements.txt (same folder)
# fastapi==0.110.2
# uvicorn[standard]==0.29.0
# httpx==0.27.0
# openai>=1.14
# ────────────────────────────────────────────────────────────────────────────
