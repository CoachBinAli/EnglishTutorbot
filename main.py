# main.py
"""
GPT Telegram bot with **voice-in & voice-out**: Whisper STT + Kokoro TTS
via Replicate, and GPT replies with OpenAI.

• Voice messages in → transcribed with Whisper (large‑v3 hash)
• GPT‑4o‑mini generates reply
• If the original message was a voice note, the reply is sent back as
  **voice** (Kokoro TTS) + text caption. If user typed text, we keep
  text‑only response.
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
# Environment
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

WHISPER_MODEL = (
    "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"
)
KOKORO_MODEL = "jaaari/kokoro-82m"  # public TTS model
DEFAULT_VOICE = "af"  # neutral female; change later if you like

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


async def send_voice(cid: int, audio_url: str, caption: str | None = None) -> None:
    payload = {"chat_id": cid, "voice": audio_url}
    if caption:
        payload["caption"] = caption
    async with httpx.AsyncClient() as hc:
        await hc.post(f"{TG_API}/sendVoice", json=payload)


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
# Replicate helpers
# ──────────────────────────────
async def transcribe(path: Path) -> str:
    def _run():
        return rep_client.run(
            WHISPER_MODEL,
            input={
                "audio": open(path, "rb"),
                "model": "large-v3",
                "language": "en",
                "transcription": "plain text",
                "temperature": 0,
            },
        )

    try:
        out = await asyncio.to_thread(_run)
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    text = out.get("transcription", "") if isinstance(out, dict) else ""
    log.info("📜 Whisper → %s", text)
    return text or "(blank transcription)"


async def synthesize(text: str, voice: str = DEFAULT_VOICE) -> str:
    def _run():
        return rep_client.run(KOKORO_MODEL, input={"text": text, "voice": voice})

    out = await asyncio.to_thread(_run)
    # Kokoro returns direct HTTPS URL string
    audio_url = out if isinstance(out, str) else out[0] if isinstance(out, list) else ""
    log.info("🔊 Kokoro audio URL: %s", audio_url)
    return audio_url


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

    is_voice = "voice" in msg and msg.get("text") is None

    # 1. Extract user prompt
    if is_voice:
        prompt_text = await transcribe(await download_voice(msg["voice"]["file_id"]))
    else:
        prompt_text = msg.get("text", "")

    if not prompt_text:
        await send_text(cid, "Sorry, I couldn't read your message.")
        return {"ok": True}

    log.info("GPT prompt: %s", prompt_text)

    try:
        reply_text = await chat(prompt_text)
    except Exception as e:
        log.exception("OpenAI error: %s", e)
        await send_text(cid, "Sorry, I'm having trouble thinking right now. Try later.")
        return {"ok": True}

    # 2. Respond
    if is_voice:
        try:
            audio_url = await synthesize(reply_text)
            await send_voice(cid, audio_url, caption=reply_text)
        except Exception as e:
            log.exception("TTS error: %s", e)
            await send_text(cid, reply_text)
    else:
        await send_text(cid, reply_text)

    return {"ok": True}


# ──────────────────────────────
# Health
# ──────────────────────────────
@app.get("/")
async def root() -> dict[str, Literal["pong"]]:
    return {"status": "pong"}

# requirements.txt remains:
# fastapi==0.110.2
# uvicorn[standard]==0.29.0
# httpx==0.27.0
# openai>=1.14
# replicate==0.24.0
