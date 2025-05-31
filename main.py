# main.py
"""
Telegram GPT tutor with **voice-in & voice-out** via Whisper + Kokoro.
Fix: Handle all Kokoro output forms (stream, URL, list) and send via
Telegram’s `sendAudio` using either an uploaded file or a direct URL.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Literal, Union

import httpx
import replicate
from fastapi import FastAPI, HTTPException, Request
from openai import AsyncOpenAI

# ──────────────────────────────
# Env vars
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

WHISPER = "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"
KOKORO = "jaaari/kokoro-82m:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13"
VOICE = "af_nicole"

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


async def send_text(cid: int, text: str) -> None:
    async with httpx.AsyncClient() as hc:
        await hc.post(f"{TG_API}/sendMessage", json={"chat_id": cid, "text": text})


async def send_audio(cid: int, audio: Union[str, Path], caption: str | None = None) -> None:
    async with httpx.AsyncClient(timeout=60) as hc:
        if isinstance(audio, Path):
            files = {"audio": (audio.name, audio.open("rb"))}
            data = {"chat_id": cid, **({"caption": caption} if caption else {})}
            await hc.post(f"{TG_API}/sendDocument", data=data, files=files)
        else:  # URL string
            payload = {"chat_id": cid, "audio": audio}
            if caption:
                payload["caption"] = caption
            await hc.post(f"{TG_API}/sendDocument", json=payload)


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
# Replicate wrappers
# ──────────────────────────────
async def transcribe(audio_path: Path) -> str:
    def _run():
        return rep_client.run(
            WHISPER,
            input={"audio": audio_path.open("rb"), "model": "large-v3", "language": "en", "transcription": "plain text", "temperature": 0},
        )

    try:
        out = await asyncio.to_thread(_run)
    finally:
        audio_path.unlink(missing_ok=True)

    transcript = out.get("transcription", "") if isinstance(out, dict) else ""
    log.info("Whisper transcript: %s", transcript)
    return transcript or "(blank)"


async def synthesize(text: str, voice: str = VOICE) -> Union[str, Path]:
    def _run():
        return rep_client.run(KOKORO, input={"text": text, "voice": voice, "speed": 1})

    out = await asyncio.to_thread(_run)

    # Cases: bytes-like stream, URL string, list[str]
    if hasattr(out, "read"):
        audio_bytes = out.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.close()
        return Path(tmp.name)
    if isinstance(out, bytes):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(out)
        tmp.close()
        return Path(tmp.name)
    if isinstance(out, str):
        return out
    if isinstance(out, list) and out and isinstance(out[0], str):
        return out[0]
    raise ValueError("Unknown Kokoro output type")


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
        await send_text(cid, "Sorry, something went wrong. Try later.")
        return {"ok": True}

    if is_voice:
        try:
            audio_ref = await synthesize(reply)
            await send_audio(cid, audio_ref, caption=reply)
            if isinstance(audio_ref, Path):
                audio_ref.unlink(missing_ok=True)
        except Exception as e:
            log.exception("TTS error: %s", e)
            await send_text(cid, reply)
    else:
        await send_text(cid, reply)

    return {"ok": True}


# ──────────────────────────────
# Health check
# ──────────────────────────────
@app.get("/")
async def root() -> dict[str, Literal["pong"]]:
    return {"status": "pong"}

# (requirements.txt unchanged)
