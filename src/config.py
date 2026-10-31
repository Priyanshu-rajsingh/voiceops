"""Configuration: environment settings and dataset discovery.

The submission must NOT contain a copy of the dataset, so the code locates it at
run time (DATASET_DIR, a nearby folder, or the original ZIP which is extracted
to a temp folder).
"""
from __future__ import annotations

import importlib.util
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

try:  # optional: load a local .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

TODAY = date(2026, 10, 20)  # fixed reference date of the dataset (KB-00.2)
REPO_ROOT = Path(__file__).resolve().parent.parent
ZIP_NAME = "Capstone_Project_Dataset_FTTT_2026.zip"
_SKIP_DIRS = {"src", "tests", "node_modules", "__pycache__", "outputs", "sample_outputs", "eval_out"}


def _is_dataset(p: Path) -> bool:
    return (p / "kb").is_dir() and (p / "data").is_dir() and (p / "tools" / "tools.py").is_file()


def _scan(base: Path, depth: int = 2) -> Path | None:
    if not base.is_dir():
        return None
    if _is_dataset(base):
        return base
    if depth == 0:
        return None
    try:
        for child in sorted(base.iterdir()):
            if child.is_dir() and not child.name.startswith(".") and child.name not in _SKIP_DIRS:
                hit = _scan(child, depth - 1)
                if hit:
                    return hit
    except PermissionError:
        pass
    return None


_DATASET: Path | None = None


def dataset_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Return the dataset root (folder containing kb/, data/, tools/)."""
    global _DATASET
    if explicit is None and _DATASET is not None:
        return _DATASET
    bases: list[Path] = []
    if explicit:
        bases.append(Path(explicit).expanduser())
    if os.getenv("DATASET_DIR"):
        bases.append(Path(os.environ["DATASET_DIR"]).expanduser())
    bases += [Path.cwd(), REPO_ROOT, REPO_ROOT.parent, Path.home() / "Downloads"]
    for base in bases:
        hit = _scan(base)
        if hit:
            _DATASET = hit.resolve()
            return _DATASET
    zips: list[Path] = []
    if os.getenv("DATASET_ZIP"):
        zips.append(Path(os.environ["DATASET_ZIP"]).expanduser())
    zips += [b / ZIP_NAME for b in (Path.cwd(), REPO_ROOT, REPO_ROOT.parent, Path.home() / "Downloads")]
    for z in zips:
        if z.is_file():
            target = Path(tempfile.gettempdir()) / "voiceops_dataset"
            if not (target.exists() and _scan(target)):
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(target)
            hit = _scan(target, depth=3)
            if hit:
                _DATASET = hit.resolve()
                return _DATASET
    raise FileNotFoundError(
        "ZentroMart dataset not found. Put the extracted 'Capstone_Project_Dataset_FTTT_2026' folder next to "
        "this repo, or set DATASET_DIR=/path/to/folder (or DATASET_ZIP=/path/to/zip) in .env."
    )


_TOOLS = None


def load_tools():
    """Load the dataset's mock tools (lookup_order, create_ticket) without touching sys.path."""
    global _TOOLS
    if _TOOLS is None:
        path = dataset_dir() / "tools" / "tools.py"
        spec = importlib.util.spec_from_file_location("zentromart_mock_tools", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _TOOLS = mod
    return _TOOLS


PROVIDER_PRESETS = {
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434/v1",
}
_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


@dataclass
class Settings:
    llm_provider: str
    llm_model: str
    llm_api_key: str
    llm_base_url: str
    top_k: int
    stt_language: str
    stt_fallbacks: list[str]
    tts_model: str
    tts_voices: dict[str, str]
    use_dense: str
    embed_model: str


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    key = os.getenv("LLM_API_KEY") or os.getenv(_KEY_ENV.get(provider, ""), "")
    base = os.getenv("LLM_BASE_URL") or PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["openai"])
    default_model = "claude-haiku-5-5" if provider == "anthropic" else ""
    return Settings(
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", default_model),
        llm_api_key=key,
        llm_base_url=base.rstrip("/"),
        top_k=int(os.getenv("TOP_K", "8")),
        stt_language=os.getenv("STT_LANGUAGE", "en-IN"),
        stt_fallbacks=[s for s in os.getenv("STT_FALLBACKS", "hi-IN").split(",") if s.strip()],
        tts_model=os.getenv("TTS_MODEL", "timbre-v2.5"),
        tts_voices={
            "en": os.getenv("TTS_VOICE_EN", "Kaveri"),
            "hi": os.getenv("TTS_VOICE_HI", "Nalini"),
            "hi-en": os.getenv("TTS_VOICE_HINGLISH", "Nalini"),
            "other": os.getenv("TTS_VOICE_OTHER", "Kaveri"),
        },
        use_dense=os.getenv("USE_DENSE", "auto").lower(),
        embed_model=os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small"),
    )
