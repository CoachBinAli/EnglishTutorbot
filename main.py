# main.py
"""
Telegram English‑tutor bot with **voice‑in & voice‑out**
======================================================
• Voice notes → Whisper (large‑v3) STT
• GPT‑4o‑mini generates reply
• If original message was voice, reply is rendered by **Kokoro‑82M** TTS
  and sent back as a WAV **document** (Telegram plays it fine). Text input
  still gets text output.
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

# ──────────────────────────── ENV ────────────────────────────
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
REPL_TOKEN = os.getenv("REPLICATE_API_TOKEN")
for k, v in {"TG_TOKEN": TG_TOKEN, "OPENAI_KEY": OPENAI_KEY, "REPL_TOKEN": REPL_TOKEN}.items():
    if not v:
        raise RuntimeError(f"Missing env var {k}")

openai_client = AsyncOpenAI(api_key=OPENAI_KEY)
rep = replicate.Client(api_token=REPL_TOKEN)

WHISPER = "openai/whisper:8099696689d249cf8b122d833c36ac3f75505c666a395ca40ef26f68e7d3d16e"
KOKORO = "jaaari/kokoro-82m:f559560eb822dc509045f3921a1921234918b91739db4bf3daab2169b71c7a13"
VOICE_ID = "af_nicole"

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger("bot")

app = FastAPI()
TG_API = f"https://api.telegram.org/bot{TG_TOKEN}"

# ─────────────────────── Telegram helpers ───────────────────────
async def tg_get(method: str, **params) -> dict:
    async with httpx.AsyncClient() as hc:
        r = await hc.get(f"{TG_API}/{method}", params=params)
        r.raise_for_status()
        return r.json()["result"]


async def send_text(cid: int, text: str) -> None:
    async with httpx.AsyncClient() as hc:
        await hc.post(f"{TG_API}/sendMessage", json={"chat_id": cid, "text": text})


async def send_audio(cid: int, audio: Union[str, Path], caption: str | None = None) -> None:
    """Send WAV as document (works for any file type)."""
    async with httpx.AsyncClient(timeout=60) as hc:
        if isinstance(audio, Path):
            files = {"document": (audio.name, audio.open("rb"))}
            data = {"chat_id": cid, **({"caption": caption} if caption else {})}
            await hc.post(f"{TG_API}/sendDocument", data=data, files=files)
        else:  # URL string from Replicate
            payload = {"chat_id": cid, "document": audio}
            if caption:
                payload["caption"] = caption
            await hc.post(f"{TG_API}/sendDocument", json=payload)


async def download_voice(file_id: str) -> Path:
    info = await tg_get("getFile", file_id=file_id)
    async with httpx.AsyncClient() as hc:
        data = await hc.get(f"https://api.telegram.org/file/bot{TG_TOKEN}/{info['file_path']}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg")
    tmp.write(data.content)
    tmp.close()
    return Path(tmp.name)

# ─────────────────────── Replicate wrappers ───────────────────────
async def transcribe(path: Path) -> str:
    def _run():
        return rep.run(WHISPER, input={"audio": path.open("rb"), "model": "large-v3", "language": "en", "transcription": "plain text", "temperature": 0})

    try:
        out = await asyncio.to_thread(_run)
    finally:
        path.unlink(missing_ok=True)

    return (out.get("transcription", "") if isinstance(out, dict) else "").strip()


async def synthesize(text: str) -> Union[str, Path]:
    def _run():
        return rep.run(KOKORO, input={"text": text, "voice": VOICE_ID, "speed": 1})

    out = await asyncio.to_thread(_run)

    # Handle stream / URL / list
    if hasattr(out, "read") or isinstance(out, bytes):
        data = out.read() if hasattr(out, "read") else out
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(data)
        tmp.close()
        return Path(tmp.name)
    if isinstance(out, str):
        return out
    if isinstance(out, list) and out and isinstance(out[0], str):
        return out[0]
    raise ValueError("Unknown Kokoro output type")

# ───────────────────────── GPT wrapper ─────────────────────────
async def chat(prompt: str) -> str:
    rsp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a friendly English tutor."}, {"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return rsp.choices[0].message.content.strip()

# ───────────────────────── Webhook ─────────────────────────
@app.api_route("/webhook/{token}", methods=["POST", "GET", "HEAD"])
async def webhook(token: str, request: Request):
    if token != TG_TOKEN:
        raise HTTPException(403)
    if request.method in ("GET", "HEAD"):
        return {"ok": True}

    upd = await request.json()
    msg = upd.get("message", {})
    cid = msg.get("chat", {}).get("id")
    if cid is None:
        return {"ok": True}

    is_voice = "voice" in msg and "text" not in msg
    prompt = msg.get("text", "")
    if is_voice:
        prompt = await transcribe(await download_voice(msg["voice"]["file_id"]))

    if not prompt:
        await send_text(cid, "Sorry, couldn't understand that.")
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

# ───────────────────────── Health ─────────────────────────
@app.get("/")
async def root() -> dict[str, Literal["pong"]]:
    return {"status": "pong"}

# requirements.txt
# fastapi==0.110.2
# uvicorn[standard]==0.29.0
# httpx==0.27.0
# openai>=1.14
# replicate==0.24.0
