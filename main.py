# main.py
"""
GPT Telegram tutor with **voice‑in & voice‑out**.

Fixes:
• Handle Kokoro TTS binary response (stream) correctly: write to temp WAV
  and upload to Telegram via **sendAudio** multipart.
• Default voice set to `af_nicole` (matches Replicate example hash).

Requirements unchanged (fastapi, uvicorn[standard], httpx, openai>=1.14, replicate==0.24.0)
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
from fastapi import FastAPI, HTTPException, Request, UploadFile
from openai import AsyncOpenAI

# ──────────────────────────────
# Environment vars
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
KOKORO_MODEL = (
    "jaaari/kokoro-82m:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13"
)
VOICE_ID = "af_nicole"

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("bot")

app = FastAPI()
TG_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ──────────────────────────────
# Telegram helpers
# ──────────────────────────────
async def tg_get(method: str, **params) -> dict:
    async with httpx.AsyncClient() as hc:
        r = await hc.get(f"{TG_API}/{method}", params=params)
        r.raise_for_status()
        return r.json()["result"]


async def tg_post_multipart(endpoint: str, files: dict, data: dict) -> None:
    async with httpx.AsyncClient() as hc:
        await hc.post(f"{TG_API}/{endpoint}", data=data, files=files, timeout=60)


async def send_text(cid: int, text: str) -> None:
    async with httpx.AsyncClient() as hc:
        await hc.post(f"{TG_API}/sendMessage", json={"chat_id": cid, "text": text})


async def send_audio(cid: int, path: Path, caption: str | None = None) -> None:
    files = {"audio": (path.name, path.open("rb"))}
    data = {"chat_id": cid}
    if caption:
        data["caption"] = caption
    await tg_post_multipart("sendAudio", files, data)


async def download_voice(file_id: str) -> Path:
    info = await tg_get("getFile", file_id=file_id)
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
async def transcribe(audio_path: Path) -> str:
    def _run():
        return rep_client.run(
            WHISPER_MODEL,
            input={"audio": audio_path.open("rb"), "model": "large-v3", "language": "en", "transcription": "plain text", "temperature": 0},
        )

    try:
        result = await asyncio.to_thread(_run)
    finally:
        audio_path.unlink(missing_ok=True)

    transcript = result.get("transcription", "") if isinstance(result, dict) else ""
    log.info("Whisper transcript: %s", transcript)
    return transcript or "(blank)"


async def synthesize(text: str, voice: str = VOICE_ID) -> Path:
    def _run():
        return rep_client.run(KOKORO_MODEL, input={"text": text, "voice": voice})

    out = await asyncio.to_thread(_run)
    # out is a binary stream-like object
    if hasattr(out, "read"):
        audio_bytes = out.read()
    elif isinstance(out, bytes):
        audio_bytes = out
    else:
        raise ValueError("Unexpected Kokoro output type")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(audio_bytes)
    tmp.close()
    return Path(tmp.name)


# ──────────────────────────────
# GPT helper
# ──────────────────────────────
async def chat(prompt: str) -> str:
    rsp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a friendly English tutor."}, {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return rsp.choices[0].message.content.strip()


# ──────────────────────────────
# Webhook
# ──────────────────────────────
@app.api_route("/webhook/{token}", methods=["POST", "GET", "HEAD"])
async def webhook(token: str, request: Request):
    if token != TELEGRAM_BOT_TOKEN:
        raise HTTPException(403)

    if request.method in ("GET", "HEAD"):
        return {"ok": True}

    update = await request.json()
    msg = update.get("message", {})
    cid = msg.get("chat", {}).get("id")
    if cid is None:
        return {"ok": True}

    is_voice = "voice" in msg and "text" not in msg
    prompt = msg.get("text", "")

    if is_voice:
        prompt = await transcribe(await download_voice(msg["voice"]["file_id"]))

    if not prompt:
        await send_text(cid, "Sorry, couldn't read your message.")
        return {"ok": True}

    log.info("Prompt: %s", prompt)

    try:
        reply = await chat(prompt)
    except Exception as e:
        log.exception("OpenAI error: %s", e)
        await send_text(cid, "Sorry, something went wrong. Try again later.")
        return {"ok": True}

    if is_voice:
        try:
            wav_path = await synthesize(reply)
            await send_audio(cid, wav_path, caption=reply)
            wav_path.unlink(missing_ok=True)
        except Exception as e:
            log.exception("TTS error: %s", e)
            await send_text(cid, reply)
    else:
        await send_text(cid, reply)

    return {"ok": True}


# ──────────────────────────────
# Health
# ──────────────────────────────
@app.get("/")
async def root() -> dict[str, Literal["pong"]]:
    return {"status": "pong"}

# (requirements.txt unchanged)
