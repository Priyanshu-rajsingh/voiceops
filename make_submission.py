#!/usr/bin/env python3
"""Check everything the brief asks for, then build  <team_name>_voiceops.zip  for the upload form.

    python make_submission.py --team my_team
Refuses to build if a required file is missing, a secret or a dataset copy would be included,
the report is longer than 4 pages, or the demo is longer than 2 minutes.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".cache", "outputs", "test_audio", ".idea", ".vscode",
             "Capstone_Project_Dataset_FTTT_2026", "dataset", ".pytest_cache"}
SKIP_FILES = {".env", ".DS_Store", "tickets_created.csv"}
DATASET_FILES = {"orders.csv", "customers.csv", "products.csv", "order_items.csv", "support_logs.csv",
                 "questions.csv", "conversations.csv", "pincodes.csv", "holidays.csv"}
KEY_PATTERNS = [r"sk-[A-Za-z0-9_\-]{20,}", r"AIza[0-9A-Za-z_\-]{30,}", r"gsk_[A-Za-z0-9]{20,}", r"sk-ant-[A-Za-z0-9_\-]{20,}"]
REQUIRED = ["README.md", ".env.example", "requirements.txt", "run.py", "src", "report.pdf", "demo.mp4", "sample_outputs"]


def files_to_pack(team_zip: str) -> list[Path]:
    out = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_dir() or any(part in SKIP_DIRS for part in rel.parts) or p.name in SKIP_FILES:
            continue
        if p.suffix == ".zip" or p.name == team_zip:
            continue
        if rel.parts[0] == "eval_out" and p.name != "summary.json":
            continue  # keep only the numbers, not row-level copies of the practice data
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name, e.g. my_team -> my_team_voiceops.zip")
    a = ap.parse_args()
    name = re.sub(r"[^A-Za-z0-9_\-]+", "_", a.team).strip("_") + "_voiceops.zip"
    errors, warns = [], []

    for r in REQUIRED:
        if not (ROOT / r).exists():
            errors.append(f"missing required item: {r}")
    sample = ROOT / "sample_outputs"
    if sample.is_dir() and not (list(sample.glob("*.json")) and list(sample.glob("*.wav"))):
        errors.append("sample_outputs/ must contain at least one .json and one .wav produced by run.py")
    rp = ROOT / "report.pdf"
    if rp.exists():
        from pypdf import PdfReader
        n = len(PdfReader(rp).pages)
        if n > 4:
            errors.append(f"report.pdf has {n} pages (max 4)")
    demo = ROOT / "demo.mp4"
    if demo.exists():
        probe = shutil.which("ffprobe")
        if probe:
            try:
                dur = float(subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of",
                                            "default=nw=1:nk=1", str(demo)], capture_output=True, text=True).stdout)
                if dur > 125:
                    errors.append(f"demo.mp4 is {dur:.0f}s (max 2 minutes)")
            except ValueError:
                warns.append("could not read demo.mp4 duration")
        if demo.stat().st_size > 200 * 1024 * 1024:
            warns.append("demo.mp4 is over 200 MB; the upload form may reject it")
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    if "github.com/" not in readme or "<your-username>" in readme:
        errors.append("README.md must contain your real public GitHub URL (replace the <your-username>/<your-repo> placeholder)")

    pack = files_to_pack(name)
    secrets = []
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                v = line.split("=", 1)[1].strip().strip("'\"")
                if len(v) >= 8:
                    secrets.append(v)
    for p in pack:
        if p.name in DATASET_FILES:
            errors.append(f"dataset copy would be included: {p.relative_to(ROOT)}")
        if p.suffix.lower() in {".py", ".md", ".txt", ".json", ".csv", ".example", ".env", ".sh", ".toml", ".yml", ".yaml"} or p.name.endswith(".example"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            if any(s in text for s in secrets) or any(re.search(k, text) for k in KEY_PATTERNS):
                errors.append(f"possible API key inside {p.relative_to(ROOT)}")
    ex = ROOT / ".env.example"
    if ex.exists():
        for line in ex.read_text(encoding="utf-8").splitlines():
            if re.match(r"^[A-Z_]+_KEY=\S{6,}", line.strip()):
                errors.append(".env.example contains a value for a *_KEY variable")

    if errors:
        print("NOT READY - fix these first:\n  - " + "\n  - ".join(errors))
        return 1
    for w in warns:
        print("warning:", w)
    out = ROOT / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in pack:
            z.write(p, p.relative_to(ROOT).as_posix())
    mb = out.stat().st_size / 1e6
    print(f"Built {out.name} ({mb:.1f} MB) with {len(pack)} files. Upload this ZIP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
