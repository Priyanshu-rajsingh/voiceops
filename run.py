#!/usr/bin/env python3
"""Batch run for the organizers: one command, a folder of customer audio in, replies out.

    python run.py --audio_dir ./test_audio --out_dir ./outputs

For every audio file this saves (in --out_dir):
    <name>.json        what Gnani STT heard, the reply text, decision, tool calls and the document
                       chunks used (retrieved + cited section IDs)
    <name>.txt         the reply text
    <name>_reply.wav   the reply spoken with Gnani TTS
and a results.csv summary of all files.

Files named like  PC003_t1.wav, PC003_t2.wav ...  are treated as ONE multi-turn conversation (shared memory);
every other file is an independent customer.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

from src import config
from src.agent import SupportAgent
from src.kb import KnowledgeBase
from src.llm import LLM
from src.voice import AUDIO_EXT, Voice, speech_text

CONV_RE = re.compile(r"^(?P<cid>.+?)[_\-](?:t|turn)[_\-]?(?P<turn>\d+)$", re.I)
UNCLEAR = "Sorry, I didn't catch that. Could you please say it again?"


def natural_key(p: Path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", p.name)]


def collect(audio_dir: Path) -> list[Path]:
    return sorted((p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT), key=natural_key)


def plan(files: list[Path], group: bool) -> list[tuple[str | None, Path]]:
    """[(conversation_id or None, file)] in processing order; turns of one conversation stay together."""
    if not group:
        return [(None, f) for f in files]
    convs: dict[str, list[tuple[int, Path]]] = {}
    single: list[Path] = []
    for f in files:
        m = CONV_RE.match(f.stem)
        if m:
            convs.setdefault(m["cid"], []).append((int(m["turn"]), f))
        else:
            single.append(f)
    ordered = [(None, f) for f in single]
    for cid, items in convs.items():
        ordered += [(cid, f) for _, f in sorted(items)]
    return ordered


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio_dir", "--audio-dir", required=True, help="folder with customer audio files")
    ap.add_argument("--out_dir", "--out-dir", default="outputs", help="where replies are saved (default: outputs)")
    ap.add_argument("--dataset", help="path to the extracted dataset folder (or set DATASET_DIR)")
    ap.add_argument("--stt-language", help="Gnani language code for STT, default from STT_LANGUAGE (en-IN)")
    ap.add_argument("--no-conversations", action="store_true", help="treat every file as an independent customer")
    ap.add_argument("--limit", type=int, help="only the first N files (quick test)")
    args = ap.parse_args(argv)

    audio_dir, out_dir = Path(args.audio_dir), Path(args.out_dir)
    if not audio_dir.is_dir():
        print(f"error: audio folder not found: {audio_dir}", file=sys.stderr)
        return 2
    files = collect(audio_dir)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"error: no audio files ({', '.join(sorted(AUDIO_EXT))}) in {audio_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        config.dataset_dir(args.dataset)
    print(f"dataset: {config.dataset_dir()}\nfiles:   {len(files)}  ->  {out_dir}/")
    agent = SupportAgent(llm=LLM(), kb=KnowledgeBase(), mode="full")
    voice = Voice()
    rows, current = [], object()

    for cid, f in plan(files, not args.no_conversations):
        if cid != current or cid is None:
            agent.reset()  # new customer
            current = cid
        t0 = time.time()
        rec: dict = {"file": f.name, "conversation_id": cid or "", "transcript": "", "stt_language": "", "language": "",
                     "decision": "", "reply_text": "", "chunks_used": [], "cited_sections": [], "tool_calls": [],
                     "ticket_id": None, "audio_file": "", "error": None}
        try:
            heard = voice.transcribe(f, args.stt_language)
            rec["transcript"], rec["stt_language"] = heard.text, heard.language_code
            if not heard.text.strip():
                rec.update(reply_text=UNCLEAR, decision="clarify", language="en", error=heard.error or "no speech recognised")
            else:
                turn = agent.respond(heard.text, channel="voice")
                d = turn.to_dict()
                rec.update(reply_text=turn.reply, decision=turn.decision, language=turn.language,
                           chunks_used=d["chunks_used"], cited_sections=d["cited_sections"], retrieved=d["retrieved"],
                           tool_calls=d["tool_calls"], ticket_id=turn.ticket_id, search_query=d["search_query"],
                           error=turn.error)
        except Exception as e:  # one bad file must never stop the batch
            rec.update(reply_text=UNCLEAR, decision="clarify", language="en", error=f"{type(e).__name__}: {e}")
        for field in ("transcript", "reply_text", "search_query"):  # never persist verification digits
            for secret in agent.state.secrets:
                if isinstance(rec.get(field), str):
                    rec[field] = rec[field].replace(secret, "****")
        spoken = speech_text(rec["reply_text"])
        wav = out_dir / f"{f.stem}_reply.wav"
        if voice.speak(spoken, rec["language"] or "en", wav):
            rec["audio_file"] = wav.name
        else:
            rec["error"] = (rec["error"] + "; " if rec["error"] else "") + "TTS failed: " + str(getattr(voice, "last_error", ""))
        rec["spoken_text"] = spoken
        rec["seconds"] = round(time.time() - t0, 2)
        (out_dir / f"{f.stem}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / f"{f.stem}.txt").write_text(rec["reply_text"] + "\n", encoding="utf-8")
        rows.append(rec)
        print(f"[{len(rows)}/{len(files)}] {f.name}: {rec['decision']:<8} {rec['seconds']:>5}s  "
              f"heard={rec['transcript'][:50]!r}" + (f"  ERROR: {rec['error']}" if rec["error"] else ""))

    with open(out_dir / "results.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "transcript", "language", "decision", "reply_text", "chunks_used", "tool_calls",
                    "ticket_id", "audio_file", "seconds", "error"])
        for r in rows:
            w.writerow([r["file"], r["transcript"], r["language"], r["decision"], r["reply_text"],
                        " ; ".join(r["chunks_used"]), " ; ".join(r["tool_calls"]), r["ticket_id"] or "",
                        r["audio_file"], r["seconds"], r["error"] or ""])
    bad = sum(1 for r in rows if r["error"])
    print(f"\nDone: {len(rows)} files, {bad} with errors. Summary: {out_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
