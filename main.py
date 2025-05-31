# main.py
"""
Text‑first GPT Telegram bot with **voice‑in (Whisper STT)** using the
**new OpenAI ≥ 1.14 SDK** and Replicate.

Flow now:
1. User sends **voice** _or_ text in Telegram.
2. If voice → download → Whisper on Replicate → transcript.
3. Transcript (or original text) → GPT‑4‑mini.
4. Bot returns **text** reply. (TTS coming next.)

Build (Render):
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000

Env vars to set in Render:
  TELEGRAM_BOT_TOKEN    # your bot token
  OPENAI_API_KEY        # OpenAI secret key
  REPLICATE_API_TOKEN   # token from https://replicate.com/account
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("bot")

app = FastAPI()

# ──────────────────────────────
# Helpers – Telegram I/O
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
# Helpers – STT via Replicate Whisper
# ──────────────────────────────
async def transcribe_voice(file_id: str) -> str:
    """Download Telegram voice message and run Whisper.
    Returns the transcribed text."""
    file_path = await telegram_get_file_path(file_id)
    audio_bytes = await telegram_download_file(file_path)

    # Save to temp OGG file
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = Path(tmp.name)

    model = "openai/whisper"
    inputs = {
        "audio": open(tmp_path, "rb"),
        "model": "large-v3",
        "language": "en",
    }

    # run Replicate in a separate thread to avoid blocking event loop
    def _run_whisper():
        return replicate_client.run(model, input=inputs)

    output = await asyncio.to_thread(_run_whisper)

    # Replicate’s Whisper returns a list of segments or dict depending on interface
    transcript = (
        output.get("transcription")
        if isinstance(output, dict)
        else " ".join(seg["text"] for seg in output if isinstance(seg, dict))
    )
    logger.info("Whisper transcript: %s", transcript)

    try:
        tmp_path.unlink(missing_ok=True)
    except OSError:
        pass

    return transcript or "(unable to transcribe)"


# ──────────────────────────────
# Helpers – GPT
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
# Webhook endpoint
# ──────────────────────────────
@app.api_route("/webhook/{token}", methods=["POST", "GET", "HEAD"])
async def telegram_webhook(token: str, request: Request):
    if token != TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token in path")

    # Handshake
    if request.method in ("GET", "HEAD"):
        return {"ok": True}

    update: dict = await request.json()
    message = update.get("message", {})
    chat_id = message.get("chat", {}).get("id")

    if chat_id is None:
        return {"ok": True}

    # Determine input text
    text: str | None = message.get("text")
    if text is None and "voice" in message:
        voice_id = message["voice"]["file_id"]
        try:
            text = await transcribe_voice(voice_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("STT failed: %s", exc)
            text = "(Sorry, I couldn't understand the audio.)"

    if text is None:
        logger.info("Ignoring unsupported message type: %s", message.keys())
        return {"ok": True}

    logger.info("Prompt for GPT: %s", text)

    try:
        reply_text = await generate_reply(text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenAI error: %s", exc)
        reply_text = (
            "Sorry, I'm having trouble generating a reply right now. Please try later."
        )

    await send_telegram_message(chat_id, reply_text)
    logger.info("Sent reply to %s", chat_id)

    return {"ok": True}


# ──────────────────────────────
# Health check
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
