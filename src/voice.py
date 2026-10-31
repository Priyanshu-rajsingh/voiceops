"""Speech layer: Gnani STT in, Gnani TTS out (the only speech services used)."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from . import config, dates

AUDIO_EXT = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".opus", ".webm"}
MAX_SECONDS = 55  # Gnani REST STT accepts up to 60 s


@dataclass
class STTResult:
    text: str = ""
    language_code: str = ""
    error: str | None = None


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _duration(path: Path) -> float | None:
    probe = shutil.which("ffprobe")
    if not probe:
        return None
    try:
        out = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of",
                              "default=nw=1:nk=1", str(path)], capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip())
    except Exception:
        return None


def _to_wav(path: Path, workdir: Path, split: bool = False) -> list[Path]:
    """Convert to 16 kHz mono WAV with ffmpeg (optionally split into 50 s pieces)."""
    ff = _ffmpeg()
    if not ff:
        return [path]
    if split:
        out = workdir / f"{path.stem}_%03d.wav"
        extra = ["-f", "segment", "-segment_time", "50"]
    else:
        out = workdir / f"{path.stem}_000.wav"
        extra = []
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000", *extra, str(out)],
                   check=True, timeout=120)
    return sorted(workdir.glob(f"{path.stem}_*.wav"))


class Voice:
    def __init__(self, settings: config.Settings | None = None):
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient

        self.s = settings or config.get_settings()
        key = os.getenv("GNANI_API_KEY")
        if not key:
            raise RuntimeError("GNANI_API_KEY is not set (see .env.example).")
        self.stt = GnaniSTTClient()  # reads GNANI_API_KEY from the environment
        self.tts = GnaniTTSClient()

    # ------------------------------------------------------------------ STT
    def _transcribe_once(self, path: Path, code: str) -> str:
        last: Exception | None = None
        for attempt in range(3):
            try:
                try:  # format=transcribe -> numbers/dates/pincodes arrive as digits (ITN)
                    res = self.stt.transcribe(str(path), language_code=code, format="transcribe")
                except TypeError:
                    res = self.stt.transcribe(str(path), language_code=code)
                return str(res.get("transcript", "") if isinstance(res, dict) else res).strip()
            except Exception as e:  # SDK raises APIError / InvalidAudioError / network errors
                last = e
                name = e.__class__.__name__
                if name in ("AuthenticationError",):
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"STT failed: {last}")

    def transcribe(self, audio_path: str | os.PathLike, language: str | None = None) -> STTResult:
        path = Path(audio_path)
        codes = [language or self.s.stt_language] + [c for c in self.s.stt_fallbacks if c != (language or self.s.stt_language)]
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            pieces = [path]
            dur = _duration(path)
            try:
                if dur and dur > MAX_SECONDS:
                    pieces = _to_wav(path, work, split=True)
            except Exception:
                pieces = [path]
            last_err = None
            for code in codes:
                texts, err = [], None
                for piece in pieces:
                    try:
                        texts.append(self._transcribe_once(piece, code))
                    except Exception as e:
                        err = str(e)
                        # unsupported container/codec -> normalise once and retry
                        if _ffmpeg() and piece == path:
                            try:
                                conv = _to_wav(path, work)[0]
                                texts.append(self._transcribe_once(conv, code))
                                err = None
                            except Exception as e2:
                                err = str(e2)
                text = " ".join(t for t in texts if t).strip()
                if text:
                    return STTResult(text, code)
                last_err = err
            return STTResult("", codes[0], last_err or "no speech recognised")

    # ------------------------------------------------------------------ TTS
    def tts_params(self, lang: str) -> tuple[str, str, str]:
        """(model, voice, language) for a reply language code."""
        v = self.s.tts_voices
        lang = (lang or "en").lower()
        if lang == "en":
            return self.s.tts_model, v["en"], "en-IN"
        if lang == "hi":
            return self.s.tts_model, v["hi"], "hi-IN"
        if lang == "hi-en":
            return self.s.tts_model, v["hi-en"], "hi-en"
        code = f"{lang.split('-')[0]}-IN"
        return os.getenv("TTS_OTHER_MODEL", "timbre-v3.0"), os.getenv("TTS_VOICE_OTHER", "Shivani"), code

    def speak(self, text: str, lang: str, out_path: str | os.PathLike) -> bool:
        model, voice, code = self.tts_params(lang)
        out = str(out_path)
        for attempt in range(3):
            try:
                audio = self.tts.synthesize(text, voice=voice, model=model, language=code, output_file=out)
                if isinstance(audio, (bytes, bytearray)) and (not os.path.exists(out) or os.path.getsize(out) == 0):
                    Path(out).write_bytes(bytes(audio))
                if os.path.exists(out) and os.path.getsize(out) > 0:
                    return True
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        return False


# ---------------------------------------------------------------------- text prepared for the ear
_MD = re.compile(r"[*_`#>]+")
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.M)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ORDER = re.compile(r"(?<![\d-])(\d{8})(?![\d-])")


def speech_text(text: str) -> str:
    """Make a written reply easy to listen to (STY-3): no markup, spoken dates, order IDs in groups of four."""
    t = _EMOJI.sub("", _BULLET.sub("", _MD.sub("", text)))
    t = _ISO.sub(lambda m: dates.spoken(date(int(m[1]), int(m[2]), int(m[3]))), t)
    t = _ORDER.sub(lambda m: f"{m[1][:4]} {m[1][4:]}", t)
    t = re.sub(r"(?<=\d)\s*[-\u2013]\s*(?=\d)", " to ", t)  # "5-7 days" -> "5 to 7 days"
    return re.sub(r"\s+", " ", t).strip()
