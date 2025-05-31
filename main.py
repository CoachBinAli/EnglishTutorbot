# main.py
"""
GPT Telegram bot with **voice‑in (Whisper STT)** using the new OpenAI
Python SDK and Replicate.  ✅ Fixed: use an explicit Whisper version ID so
Replicate doesn’t 404.

Flow ➜ voice OGG → Whisper (large‑v3) → text → GPT‑4‑mini → text reply.

Build (Render):
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000

Env vars (Render ➜ Environment):
  TELEGRAM_BOT_TOKEN   # Telegram bot token
  OPENAI_API_KEY       # OpenAI secret key
  REPLICATE_API_TOKEN  # from https://replicate.com/account
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

import httpx
import replicate
from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI

# ──────────────────────────────
# Environment & clients
# ──────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")

for key, val in {
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "REPLICATE_API_TOKEN": REPLICATE_TOKEN,
}.items():
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
replicate_client = replicate.Client(api_token=REPLICATE_TOKEN)

# Explicit version tag avoids 404
WHISPER_VERSION = "openai/whisper:20231130"  # latest public large-v3

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("bot")

app = FastAPI()

# ──────────────────────────────
# Telegram helpers
# ──────────────────────────────
async def send_telegram_message(chat_id: int, text: str) -> None:
    async with httpx.AsyncClient(timeout=15) as hc:
        await hc.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )


async def telegram_get_file_path(file_id: str) -> str:
    async with httpx.AsyncClient(timeout=15) as hc:
        r = await hc.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile",
            params={"file_id": file_id},
        )
        r.raise_for_status()
        return r.json()["result"]["file_path"]


async def telegram_download_file(file_path: str) -> bytes:
    async with httpx.AsyncClient(timeout=60) as hc:
        r = await hc.get(
            f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
        )
        r.raise_for_status()
        return r.content


# ──────────────────────────────
# STT helper
# ──────────────────────────────
async def transcribe_voice(file_id: str) -> str:
    """Download voice message and return Whisper transcription."""
    try:
        file_path = await telegram_get_file_path(file_id)
        raw = await telegram_download_file(file_path)
    except Exception as exc:
        logger.exception("Telegram file download failed: %s", exc)
        return "(audio download error)"

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    def _run_whisper() -> dict | list:
        return replicate_client.run(
            WHISPER_VERSION,
            input={
                "audio": open(tmp_path, "rb"),
                "model": "large-v3",
                "language": "en",
            },
        )

    try:
        output = await asyncio.to_thread(_run_whisper)
    except Exception as exc:
        logger.exception("Replicate Whisper failed: %s", exc)
        return "(speech‑to‑text error)"
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    if isinstance(output, dict):
        transcript = output.get("transcription", "")
    else:  # list of segments
        transcript = " ".join(seg.get("text", "") for seg in output)

    logger.info("Whisper transcript ➜ %s", transcript)
    return transcript or "(blank transcription)"


# ──────────────────────────────
# GPT helper
# ──────────────────────────────
async def generate_reply(prompt: str) -> str:
    resp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly English tutor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# ──────────────────────────────
# Webhook
# ──────────────────────────────
@app.api_route("/webhook/{token}", methods=["POST", "GET", "HEAD"])
async def telegram_webhook(token: str, request: Request):
    if token != TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token in path")

    if request.method in ("GET", "HEAD"):
        return {"ok": True}

    update = await request.json()
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    if chat_id is None:
        return {"ok": True}

    text: str | None = message.get("text")
    if text is None and "voice" in message:
        text = await transcribe_voice(message["voice"]["file_id"])

    if text is None:
        logger.info("Unsupported message type: keys=%s", message.keys())
        return {"ok": True}

    logger.info("GPT prompt: %s", text)

    try:
        reply = await generate_reply(text)
    except Exception as exc:
        logger.exception("OpenAI error: %s", exc)
        reply = "Sorry, I'm having trouble thinking right now. Please try again later."

    await send_telegram_message(chat_id, reply)
    logger.info("Sent reply to %s", chat_id)

    return {"ok": True}


# ──────────────────────────────
# Health
# ──────────────────────────────
@app.get("/")
async def root() -> dict[str, Literal["pong"]]:
    return {"status": "pong"}

# ──────────────────────────────
# requirements.txt (same folder)
# fastapi==0.110.2
# uvicorn[standard]==0.29.0
# httpx==0.27.0
# openai>=1.14
# replicate==0.24.0
# ──────────────────────────────
