# main.py
"""
GPT Telegram bot with **voice‑in** (Whisper STT on Replicate) and the new
OpenAI SDK.

✔ Whisper pinned to large‑v3 hash so it never 404s.
✔ Pass explicit `transcription="plain text"` and other recommended params
  for consistent output.
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
# Env
# ──────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN")

for k, v in {
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "REPLICATE_API_TOKEN": REPLICATE_TOKEN,
}.items():
    if not v:
        raise RuntimeError(f"Missing env var: {k}")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
rep_client = replicate.Client(api_token=REPLICATE_TOKEN)

WHISPER_HASH = (
    "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("bot")

app = FastAPI()
TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ──────────────────────────────
# Telegram helpers
# ──────────────────────────────
async def tg_call(method: str, **params) -> dict:
    async with httpx.AsyncClient() as hc:
        r = await hc.get(f"{TG_API}/{method}", params=params)
        r.raise_for_status()
        return r.json()["result"]


async def send_text(cid: int, text: str) -> None:
    async with httpx.AsyncClient() as hc:
        await hc.post(f"{TG_API}/sendMessage", json={"chat_id": cid, "text": text})


async def download_voice(file_id: str) -> Path:
    info = await tg_call("getFile", file_id=file_id)
    async with httpx.AsyncClient() as hc:
        data = await hc.get(
            f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{info['file_path']}"
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    tmp.write(data.content)
    tmp.close()
    return Path(tmp.name)


# ──────────────────────────────
# Whisper STT
# ──────────────────────────────
async def transcribe(path: Path) -> str:
    def _run():
        audio_url = rep_client.files.upload(path)
        return rep_client.run(
            WHISPER_HASH,
            input={
                "audio": audio_url,
                "model": "large-v3",
                "language": "en",
                "transcription": "plain text",
                "translate": False,
                "temperature": 0,
                "suppress_tokens": "-1",
                "logprob_threshold": -1,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": True,
                "compression_ratio_threshold": 2.4,
                "temperature_increment_on_fallback": 0.2,
            },
        )

    try:
        out = await asyncio.to_thread(_run)
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    transcript = out.get("transcription", "") if isinstance(out, dict) else ""
    log.info("Whisper transcript → %s", transcript)
    return transcript or "(blank transcription)"


# ──────────────────────────────
# GPT helper
# ──────────────────────────────
async def chat(prompt: str) -> str:
    rsp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly English tutor."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return rsp.choices[0].message.content.strip()


# ──────────────────────────────
# Webhook
# ──────────────────────────────
@app.api_route("/webhook/{token}", methods=["POST", "GET", "HEAD"])
async def webhook(token: str, request: Request):
    if token != TELEGRAM_BOT_TOKEN:
        raise HTTPException(403, "Bad token")

    if request.method in ("GET", "HEAD"):
        return {"ok": True}

    data = await request.json()
    msg = data.get("message", {})
    cid = msg.get("chat", {}).get("id")
    if cid is None:
        return {"ok": True}

    text = msg.get("text")
    if text is None and "voice" in msg:
        voice_path = await download_voice(msg["voice"]["file_id"])
        text = await transcribe(voice_path)

    if text is None:
        log.info("Unsupported message")
        return {"ok": True}

    log.info("GPT prompt: %s", text)

    try:
        reply = await chat(text)
    except Exception as e:
        log.exception("OpenAI error: %s", e)
        reply = "Sorry, I'm having trouble thinking right now. Please try again later."

    await send_text(cid, reply)
    return {"ok": True}


# ──────────────────────────────
# Health
# ──────────────────────────────
@app.get("/")
async def root() -> dict[str, Literal["pong"]]:
    return {"status": "pong"}

# requirements.txt unchanged
