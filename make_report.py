#!/usr/bin/env python3
"""Build report.pdf (max 4 pages) from eval_out/summary.json + report_notes.md.

    python make_report.py            # after:  python -m src.evaluate
Every number in the PDF is read from your own evaluation run - nothing is typed in by hand.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent


def pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.0f}%"


def notes(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    out, cur = {}, None
    for line in text.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip().lower()
            out[cur] = ""
        elif cur is not None:
            out[cur] += line + "\n"
    return {k: v.strip() for k, v in out.items()}


def bullets(block: str) -> list[str]:
    return [l.lstrip("-* ").strip() for l in block.splitlines() if l.strip().startswith(("-", "*"))]


def pipeline_drawing(width: float) -> Drawing:
    d = Drawing(width, 2.7 * cm)
    boxes = [["Customer audio", "Gnani STT", "Router (LLM)", "Identity gate + tools"],
             ["Hybrid retriever", "Answer (LLM)", "Guardrails + ticket", "Gnani TTS"]]
    bw, bh, gap = 3.6 * cm, 0.9 * cm, (width - 4 * 3.6 * cm) / 3
    for r, row in enumerate(boxes):
        y = (1.6 if r == 0 else 0.1) * cm
        for i, label in enumerate(row):
            x = i * (bw + gap)
            dark = label.startswith(("Router", "Answer"))
            d.add(Rect(x, y, bw, bh, rx=4, ry=4, fillColor=colors.HexColor("#243040" if dark else "#F4F1EC"),
                       strokeColor=colors.HexColor("#8A8F98"), strokeWidth=0.6))
            d.add(String(x + bw / 2, y + 0.33 * cm, label, textAnchor="middle", fontSize=7.5,
                         fillColor=colors.white if dark else colors.HexColor("#243040")))
            if i < 3:
                ax = x + bw
                d.add(Line(ax, y + bh / 2, ax + gap - 3, y + bh / 2, strokeColor=colors.HexColor("#555")))
                d.add(Polygon([ax + gap - 3, y + bh / 2 + 2, ax + gap, y + bh / 2, ax + gap - 3, y + bh / 2 - 2],
                              fillColor=colors.HexColor("#555")))
    x = 3 * (bw + gap) + bw / 2  # turn the arrow from the end of row 1 to the start of row 2
    d.add(Line(x, 1.6 * cm, x, 1.35 * cm, strokeColor=colors.HexColor("#555")))
    d.add(Line(x, 1.35 * cm, bw / 2, 1.35 * cm, strokeColor=colors.HexColor("#555")))
    d.add(Line(bw / 2, 1.35 * cm, bw / 2, 1.0 * cm, strokeColor=colors.HexColor("#555")))
    d.add(Polygon([bw / 2 - 2, 1.04 * cm + 3, bw / 2, 1.0 * cm, bw / 2 + 2, 1.04 * cm + 3], fillColor=colors.HexColor("#555")))
    return d


def chart(summary: dict, path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    steps = [("Step 1: prompt only", "step1", "#B8B8C0"), ("Step 2: + RAG", "step2", "#7B8CDE"),
             ("Step 3: + memory/tools", "step3_questions", "#243040")]
    metrics = [("Overall pass", "pass_rate"), ("Right decision", "decision_acc"), ("Required facts", "fact_rate")]
    fig, ax = plt.subplots(figsize=(7.2, 2.3), dpi=200)
    w = 0.26
    for j, (label, key, col) in enumerate(steps):
        if key in summary:
            vals = [100 * summary[key].get(m, 0) for _, m in metrics]
            bars = ax.bar([i + (j - 1) * w for i in range(len(metrics))], vals, w, label=label, color=col)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=6)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([n for n, _ in metrics], fontsize=7)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of 52 practice questions", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=6, ncol=3, loc="upper center", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def failure_text(summary: dict, eval_dir: Path) -> list[str]:
    out = []
    key = "step3_questions" if "step3_questions" in summary else "step2"
    if key in summary:
        weak = sorted(((v["pass_rate"], k, v["n"]) for k, v in summary[key]["by_intent"].items() if v["n"] >= 2))[:3]
        if weak:
            out.append("Weakest question types (single questions, final agent): " +
                       "; ".join(f"{k} ({pct(p)} of {n})" for p, k, n in weak) + ".")
        lang = summary[key]["by_language"]
        out.append("Pass rate by customer language: " + ", ".join(f"{k} {pct(v['pass_rate'])} (n={v['n']})" for k, v in lang.items()) + ".")
        f = eval_dir / f"{key}.csv"
        if f.exists():
            rows = list(csv.DictReader(open(f, encoding="utf-8")))
            bad = [r for r in rows if r["pass"] != "True"]
            causes = {"wrong decision": sum(r["decision_ok"] != "True" for r in bad),
                      "missing required fact": sum(r["include_ok"] != "True" for r in bad),
                      "forbidden claim": sum(r["not_ok"] != "True" for r in bad)}
            out.append(f"{len(bad)} of {len(rows)} questions failed; causes (a question can have several): " +
                       ", ".join(f"{k} {v}" for k, v in causes.items()) + ".")
            ex = [f"{r['id']} (needed: {r['missing'][:60]})" for r in bad if r["missing"]][:3]
            if ex:
                out.append("Examples where a required fact was missing: " + "; ".join(ex) + ".")
    if "conversations" in summary:
        weak = sorted(((v["pass_rate"], k, v["n"]) for k, v in summary["conversations"]["by_scenario"].items()))[:3]
        out.append("Weakest conversation scenarios: " + "; ".join(f"{k} ({pct(p)} of {n} turns)" for p, k, n in weak) + ".")
    return out


def build(summary_path: Path, notes_path: Path, out: Path, allow_placeholders: bool, font: float) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    nt = notes(notes_path)
    team = nt.get("team", "")
    if not allow_placeholders and re.search(r"<[^>]+>", team + nt.get("what went wrong and what i would fix", "")):
        sys.exit("report_notes.md still contains <placeholders>. Fill in the Team block and the 'What went wrong' "
                 "bullets (or use --allow-placeholders for a draft).")
    meta = summary.get("meta", {})
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=font, leading=font * 1.32, spaceAfter=3)
    small = ParagraphStyle("s", parent=body, fontSize=font - 1.5, leading=(font - 1.5) * 1.3, textColor=colors.HexColor("#444"))
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=15, leading=18, spaceAfter=2, alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=font + 2, leading=font + 5, spaceBefore=6, spaceAfter=2,
                        textColor=colors.HexColor("#243040"))
    bl = ParagraphStyle("bl", parent=body, leftIndent=10, bulletIndent=0)
    W = A4[0] - 3.2 * cm
    F = []
    F.append(Paragraph("VoiceOps 2026: ZentroMart voice support agent", h1))
    team_line = " | ".join(l.strip() for l in team.splitlines() if l.strip())
    F.append(Paragraph(escape(team_line), small))
    F.append(Paragraph(f"LLM: {meta.get('provider', '?')} / {meta.get('model', '?')} | retrievers: "
                       f"{' + '.join(meta.get('retrievers', []))} | top-k = {meta.get('top_k', '?')} | speech: Gnani STT and TTS only | "
                       f"dataset date fixed at 20 October 2026", small))

    F.append(Paragraph("1. What I built, step by step", h2))
    F.append(pipeline_drawing(W))
    F.append(Spacer(1, 3))
    steps = [
        ("Step 1: prompt only.", "A system prompt gives the assistant its ZentroMart role, limits (cannot see orders, never promise "
         "refunds), the rule to say it is an AI assistant in its first reply, and a JSON answer format "
         "(decision + reply). It sees no company documents, so it guesses. This is the baseline."),
        ("Step 2: grounded in the documents.", "Each Markdown section is one chunk that keeps its [SECTION-ID], status, type and "
         "effective date; catalog rows are chunks too. An LLM first rewrites the customer's words (Hindi, Hinglish or English, "
         "or speech-to-text) into an English policy query. Retrieval fuses BM25 and character n-gram TF-IDF "
         "(plus dense multilingual embeddings when installed) with reciprocal-rank fusion, then applies the precedence rule "
         "(current policy > agent guide > help > FAQ > marketing; superseded documents down-weighted). Products named by SKU or "
         "title pull in their return category, and the rule-of-precedence and business-day sections are added when sources overlap. "
         "The answer prompt allows only the retrieved text, requires 'I do not have that information' plus an offer of a human "
         "when nothing covers the question, and every reply records the chunks it used."),
        ("Step 3: a real conversation.", "The session remembers the order, the verification state and the last turns, so "
         "follow-ups work. Plain Python, not the model, enforces identity: an order ID triggers an unverified lookup, one value "
         "(last four phone digits or pincode) is checked, one retry is allowed, and after a second failure no order data is shared. "
         "Date arithmetic is precomputed (business days skip Sundays and listed holidays) and given to the model as a helper. "
         "Escalation rules (damage over 10,000 rupees, third contact in 7 days, human requested, legal, safety, unauthorised "
         "activity) create a ticket, and the reply claims a ticket only if the tool returned created=True. Verification digits are "
         "redacted from tickets and logs."),
        ("Step 4: voice.", "Customer audio goes to Gnani STT (en-IN by default, which accepts English mixed with Hindi; hi-IN as "
         "fallback; digits via format=transcribe; clips over 55 s are split). The reply is shortened for listening "
         "(dates spoken with the year, order IDs in groups of four, no markup) and spoken with Gnani TTS, choosing voice and language "
         "per reply. Unclear or empty audio gets a polite 'please say it again'. run.py processes a folder of audio and saves text, "
         "audio and the chunks used; app.py is a microphone demo.")]
    for head, txt in steps:
        F.append(Paragraph(f"<b>{head}</b> {txt}", body))

    F.append(Paragraph("2. Results on the practice set (my own evaluation)", h2))
    s1, s2, s3 = summary.get("step1", {}), summary.get("step2", {}), summary.get("step3_questions", {})
    r2, r3 = s2.get("retrieval", {}), s3.get("retrieval", {})
    k = r2.get("k") or r3.get("k") or meta.get("top_k", "k")
    rows = [["Metric (52 practice questions)", "Step 1", "Step 2", "Step 3"],
            ["Overall pass (right decision + facts + nothing forbidden)", pct(s1.get("pass_rate")), pct(s2.get("pass_rate")), pct(s3.get("pass_rate"))],
            ["Decision matches expected", pct(s1.get("decision_acc")), pct(s2.get("decision_acc")), pct(s3.get("decision_acc"))],
            ["Required facts present", pct(s1.get("fact_rate")), pct(s2.get("fact_rate")), pct(s3.get("fact_rate"))],
            ["No forbidden claim", pct(s1.get("safe_rate")), pct(s2.get("safe_rate")), pct(s3.get("safe_rate"))],
            [f"Retrieval: gold section found in top results (hit@{k})", "n/a", pct(r2.get("hit")), pct(r3.get("hit"))],
            [f"Retrieval: share of gold sections found (recall@{k})", "n/a", pct(r2.get("recall")), pct(r3.get("recall"))]]
    t = Table(rows, colWidths=[W - 3 * 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), font - 1), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243040")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#BBB")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F5F2")]),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    F.append(t)
    F.append(Spacer(1, 3))
    png = ROOT / "eval_out" / "chart.png"
    png.parent.mkdir(exist_ok=True)
    if chart(summary, png):
        F.append(Image(str(png), width=W, height=W * 2.3 / 7.2))
    delta = (s2.get("pass_rate", 0) - s1.get("pass_rate", 0)) * 100
    c = summary.get("conversations", {})
    F.append(Paragraph(
        f"<b>Step 2 vs Step 1:</b> grounding the model in the documents changed the overall pass rate from {pct(s1.get('pass_rate'))} "
        f"to {pct(s2.get('pass_rate'))} ({delta:+.0f} points). "
        + (f"<b>Conversations</b> ({c.get('conversations', '?')} dialogues, {c.get('turns', '?')} turns): "
           f"{pct(c.get('turn_pass_rate'))} of turns pass, {pct(c.get('conversation_pass_rate'))} of whole conversations pass, "
           f"decision accuracy {pct(c.get('decision_acc'))}, expected tool calls made correctly {pct(c.get('tool_acc'))}. " if c else "")
        + "A question passes only if the decision is acceptable, every required fact appears (dates, numbers and Hindi digits "
        "accepted in any equivalent form) and no forbidden phrase appears.", body))
    off = summary.get("retrieval_offline", {}).get("by_k", {}).get(str(k))
    if off and r2:
        F.append(Paragraph(f"<b>Effect of the LLM query rewrite on retrieval:</b> searching with the raw customer text alone finds a gold "
                           f"section in the top {k} for {pct(off['hit'])} of questions (recall {pct(off['recall'])}); with the rewrite "
                           f"it is {pct(r2.get('hit'))} (recall {pct(r2.get('recall'))}).", body))

    F.append(Paragraph("3. What went wrong and what I would fix", h2))
    for line in failure_text(summary, ROOT / "eval_out"):
        F.append(Paragraph(line, bl, bulletText="•"))
    for b in bullets(nt.get("what went wrong and what i would fix", "")):
        F.append(Paragraph(escape(b), bl, bulletText="•"))
    F.append(Paragraph("Limits of my evaluation: it has 52 questions and 23 conversations; scoring is string matching, so a "
                       "paraphrased forbidden claim or a correct answer in unexpected wording can be mis-scored; the identity check "
                       "is a synthetic teaching mock, not real authentication.", small))
    bonus = bullets(nt.get("bonus work", ""))
    if bonus and not (len(bonus) == 1 and bonus[0].lower().startswith("none")):
        F.append(Paragraph("4. Bonus work", h2))
        for b in bonus:
            F.append(Paragraph(escape(b), bl, bulletText="•"))
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.4 * cm,
                            bottomMargin=1.3 * cm, title="VoiceOps 2026 report", author=team_line[:80])
    doc.build(F)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=str(ROOT / "eval_out" / "summary.json"))
    ap.add_argument("--notes", default=str(ROOT / "report_notes.md"))
    ap.add_argument("--out", default=str(ROOT / "report.pdf"))
    ap.add_argument("--allow-placeholders", action="store_true")
    ap.add_argument("--allow-partial", action="store_true", help="draft from a partial evaluation (never submit)")
    a = ap.parse_args()
    sp = Path(a.summary)
    if not sp.exists():
        sys.exit("No evaluation results found. Run:  python -m src.evaluate   first.")
    s = json.loads(sp.read_text(encoding="utf-8"))
    missing = [k for k in ("step1", "step2", "step3_questions", "conversations") if k not in s]
    if missing:
        sys.exit(f"Evaluation results are incomplete (missing: {', '.join(missing)}). Run:  python -m src.evaluate")
    short = [k for k in ("step1", "step2", "step3_questions") if s[k].get("n", 0) < 52]
    if s["conversations"].get("turns", 0) < 96:
        short.append("conversations")
    if short and not a.allow_partial:
        sys.exit("These results come from a partial run (" + ", ".join(short) + "): the practice set has 52 questions and "
                 "96 conversation turns. Finish the full evaluation first:  python -m src.evaluate --workers 1")
    from pypdf import PdfReader
    for font in (8.6, 8.2, 7.8, 7.4):
        build(sp, Path(a.notes), Path(a.out), a.allow_placeholders, font)
        pages = len(PdfReader(a.out).pages)
        if pages <= 4:
            print(f"Wrote {a.out}: {pages} page(s), body font {font} pt")
            return 0
    sys.exit(f"Report is {pages} pages (max 4). Shorten report_notes.md.")


if __name__ == "__main__":
    sys.exit(main())
