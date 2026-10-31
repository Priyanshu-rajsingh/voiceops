"""Evaluate the agent on the practice files and write numbers for the report.

  python -m src.evaluate --retrieval-only          # no LLM needed: retrieval quality of the raw question
  python -m src.evaluate --steps 1 2 3 conv        # full evaluation (needs LLM key); default = all
  options: --asr (use the speech-to-text style question text), --limit N, --workers 4

Writes eval_out/summary.json (+ per-row CSVs). Step 1 = prompt only, Step 2 = RAG,
Step 3 = RAG + memory + tools (single questions), conv = multi-turn practice conversations.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from . import config, scoring
from .agent import SupportAgent
from .kb import KnowledgeBase
from .llm import LLM, DailyLimitError

OUT = config.REPO_ROOT / "eval_out"


def read_csv(name: str) -> list[dict]:
    with open(config.dataset_dir() / "practice" / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _rate(xs: list[bool]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def _group(rows: list[dict], key: str, field: str = "pass") -> dict:
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(bool(r[field]))
    return {k: {"n": len(v), "pass_rate": _rate(v)} for k, v in sorted(g.items())}


def _write(path: Path, rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    if rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def retrieval_summary(rows: list[dict], k: int) -> dict:
    m = [r["_ret"] for r in rows if r.get("_ret")]
    if not m:
        return {}
    return {"k": k, "n": len(m), "recall": round(statistics.mean(x["recall"] for x in m), 4),
            "hit": _rate([x["hit"] for x in m]), "mrr": round(statistics.mean(x["mrr"] for x in m), 4),
            "hard_negative_first": _rate([x["hard_neg_first"] for x in m])}


# ---------------------------------------------------------------------------- single questions
def run_questions(mode: str, kb, llm, text_col: str, limit: int | None, workers: int) -> tuple[list[dict], dict]:
    qs = read_csv("questions.csv")[: limit or None]
    k = config.get_settings().top_k

    def one(q: dict) -> dict:
        agent = SupportAgent(llm=llm, kb=kb, mode=mode)
        t = agent.respond(q[text_col], channel="chat")
        ok_inc, missing = scoring.check_must_include(t.reply, q["must_include"])
        ok_not, bad = scoring.check_must_not(t.reply, q["must_not_include"])
        ok_dec = scoring.decision_ok(t.decision, q["expected_decision"], q["acceptable_decisions"])
        ret = scoring.retrieval_metrics(q["gold_evidence"], t.chunks_used, q["hard_negatives"]) if mode != "baseline" else None
        print(f"  [{mode}] {q['id']:<7} {'PASS' if (ok_dec and ok_inc and ok_not) else 'fail'}  {t.latency:5.1f}s", flush=True)
        return {"id": q["id"], "language": q["language"], "intent": q["intent"], "expected": q["expected_decision"],
                "decision": t.decision, "decision_ok": ok_dec, "include_ok": ok_inc, "not_ok": ok_not,
                "pass": ok_dec and ok_inc and ok_not, "missing": " ; ".join(missing), "forbidden": " ; ".join(bad),
                "reply": t.reply, "chunks_used": " ; ".join(t.chunks_used), "error": t.error or "", "_ret": ret}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, qs))
    summary = {"n": len(rows), "pass_rate": _rate([r["pass"] for r in rows]),
               "decision_acc": _rate([r["decision_ok"] for r in rows]),
               "fact_rate": _rate([r["include_ok"] for r in rows]),
               "safe_rate": _rate([r["not_ok"] for r in rows]),
               "by_language": _group(rows, "language"), "by_intent": _group(rows, "intent"),
               "errors": sum(1 for r in rows if r["error"])}
    if mode != "baseline":
        summary["retrieval"] = retrieval_summary(rows, k)
    return rows, summary


# ---------------------------------------------------------------------------- conversations
def run_conversations(kb, llm, limit: int | None, workers: int) -> tuple[list[dict], dict]:
    turns = read_csv("conversations.csv")
    convs: dict[str, list[dict]] = defaultdict(list)
    for r in turns:
        convs[r["conversation_id"]].append(r)
    ids = sorted(convs)[: limit or None]

    def one(cid: str) -> list[dict]:
        agent = SupportAgent(llm=llm, kb=kb, mode="full")
        out = []
        for r in sorted(convs[cid], key=lambda x: int(x["turn"])):
            t = agent.respond(r["customer_message"], channel="chat")
            ok_inc, missing = scoring.check_must_include(t.reply, r["must_include"])
            ok_not, bad = scoring.check_must_not(t.reply, r["must_not_include"])
            ok_dec = t.decision == r["expected_decision"]
            ok_tool = scoring.tools_ok(r["expected_tool_calls"], t.tool_calls)
            ret = scoring.retrieval_metrics(r["gold_evidence"], t.chunks_used)
            print(f"  [conv] {cid} turn {r['turn']:<2} {'PASS' if (ok_dec and ok_inc and ok_not and ok_tool) else 'fail'}  {t.latency:5.1f}s", flush=True)
            out.append({"conversation_id": cid, "turn": r["turn"], "scenario": r["scenario"], "language": r["language"],
                        "expected": r["expected_decision"], "decision": t.decision, "decision_ok": ok_dec,
                        "include_ok": ok_inc, "not_ok": ok_not, "tools_ok": ok_tool,
                        "pass": ok_dec and ok_inc and ok_not and ok_tool, "missing": " ; ".join(missing),
                        "forbidden": " ; ".join(bad), "customer": r["customer_message"], "reply": t.reply,
                        "tools_called": " ; ".join(t.tool_call_strings(mask=True)),
                        "tools_expected": _mask(r["expected_tool_calls"]),
                        "chunks_used": " ; ".join(t.chunks_used), "error": t.error or "", "_ret": ret})
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = [r for part in ex.map(one, ids) for r in part]
    conv_pass = defaultdict(list)
    for r in rows:
        conv_pass[r["conversation_id"]].append(r["pass"])
    summary = {"turns": len(rows), "conversations": len(conv_pass),
               "turn_pass_rate": _rate([r["pass"] for r in rows]),
               "conversation_pass_rate": _rate([all(v) for v in conv_pass.values()]),
               "decision_acc": _rate([r["decision_ok"] for r in rows]),
               "tool_acc": _rate([r["tools_ok"] for r in rows]),
               "fact_rate": _rate([r["include_ok"] for r in rows]),
               "safe_rate": _rate([r["not_ok"] for r in rows]),
               "by_scenario": _group(rows, "scenario"),
               "retrieval": retrieval_summary(rows, config.get_settings().top_k), "errors": sum(1 for r in rows if r["error"])}
    return rows, summary


def _mask(spec: str) -> str:
    import re
    return re.sub(r"verification_value=\w+", "verification_value=****", spec)


# ---------------------------------------------------------------------------- offline retrieval check
def retrieval_only(text_col: str, ks=(4, 6, 8)) -> dict:
    kb = KnowledgeBase()
    qs = read_csv("questions.csv")
    out = {"retrievers": kb.retrievers, "query": "raw customer text (no LLM rewrite)", "by_k": {}}
    for k in ks:
        rows = []
        for q in qs:
            ids = [h.chunk.id for h in kb.search([q[text_col]], k=k)]
            rows.append({"_ret": scoring.retrieval_metrics(q["gold_evidence"], ids, q["hard_negatives"])})
        out["by_k"][str(k)] = retrieval_summary(rows, k)
    return out


# ---------------------------------------------------------------------------- CLI
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="*", default=["1", "2", "3", "conv"], choices=["1", "2", "3", "conv"])
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--asr", action="store_true", help="use asr_text instead of question")
    ap.add_argument("--limit", type=int, help="only the first N practice questions")
    ap.add_argument("--conv-limit", type=int, help="only the first N conversations (default: same as --limit)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dataset")
    args = ap.parse_args()
    if args.dataset:
        config.dataset_dir(args.dataset)
    col = "asr_text" if args.asr else "question"
    OUT.mkdir(exist_ok=True)
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}

    if args.retrieval_only:
        summary["retrieval_offline"] = retrieval_only(col)
        print(json.dumps(summary["retrieval_offline"], indent=2))
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    llm = LLM()
    kb = KnowledgeBase()
    s = config.get_settings()
    summary["meta"] = {"provider": llm.provider, "model": llm.model, "retrievers": kb.retrievers, "top_k": s.top_k,
                       "question_text": col, "date": datetime.now().isoformat(timespec="seconds")}
    plan = {"1": ("baseline", "step1"), "2": ("rag", "step2"), "3": ("full", "step3_questions")}
    for step in args.steps:
        t0 = time.time()
        try:
            if step == "conv":
                rows, summ = run_conversations(kb, llm, args.conv_limit or args.limit, args.workers)
                name = "conversations"
            else:
                mode, name = plan[step]
                rows, summ = run_questions(mode, kb, llm, col, args.limit, args.workers)
        except DailyLimitError as e:
            left = " ".join(args.steps[args.steps.index(step):])
            print(f"\nSTOPPED at step {step}: your LLM provider's DAILY token limit is used up.\n  {str(e)[:260]}\n"
                  f"Nothing was saved for this step. Wait until the limit resets (see 'try again in ...'), or switch "
                  f"LLM_MODEL in .env, then continue with:\n  python -m src.evaluate --steps {left} --limit {args.limit or 52}")
            return
        summ["seconds"] = round(time.time() - t0, 1)
        summary[name] = summ
        _write(OUT / f"{name}.csv", [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
        head = {k: v for k, v in summ.items() if not isinstance(v, dict)}
        print(f"[{name}] {json.dumps(head)}")
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
