"""Offline tests (no API keys needed): python -m tests.test_offline

They exercise everything that must NOT depend on model quality: identity gate, retry limit,
tool calls, ticket honesty, secret redaction, date maths, scoring and retrieval.
A scripted stand-in replaces the LLM, so these tests check the plumbing, not the model.
"""
from __future__ import annotations

import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, dates, scoring
from src.agent import SupportAgent, extract_order_id, extract_verification
from src.kb import KnowledgeBase


class ScriptedLLM:
    """Pretends to be the LLM: reads the context the agent builds and answers by simple rules."""

    provider, model, calls = "scripted", "scripted", 0

    def chat_json(self, system, user, **kw):
        if "understanding module" in system:
            msg = user.split("NEWEST customer message:\n", 1)[1].split("\n\nToday is", 1)[0]
            low = msg.lower()
            return {"language": "hi-en" if "hai" in low else "en", "english_query": msg, "search_terms": [],
                    "order_id": None, "verification_value": None, "dates": [],
                    "states_issue": "teesri" in low or "late" in low,
                    "wants_human": "human" in low, "legal_threat": "court" in low, "safety_risk": False,
                    "unauthorised_activity": False, "issue_summary": msg}
        ctx = user
        intro = "I'm ZentroMart's AI support assistant. " if "FIRST REPLY: yes" in ctx else ""
        v = re.search(r"VERIFICATION: (.*)", ctx).group(1)
        newest = ctx.split("=== NEWEST CUSTOMER MESSAGE ===\n", 1)[1]
        if v.startswith("NOT STARTED"):
            return {"decision": "clarify", "reply": intro + "Could you share your order ID?", "cited_sections": ["STY-1"]}
        if v.startswith("PENDING"):
            return {"decision": "clarify", "reply": intro + "Please share the last four digits of your phone or the pincode.", "cited_sections": ["IDV-2"]}
        if v.startswith("FAILED ONCE"):
            return {"decision": "clarify", "reply": "That does not match. Please try again.", "cited_sections": ["IDV-3"]}
        if v.startswith("FAILED TWICE"):
            return {"decision": "refuse", "reply": "I can't share order details; they do not match the registered ones.", "cited_sections": ["IDV-3"]}
        if "teesri" in newest.lower() and "THIRD or later" in ctx:
            return {"decision": "escalate", "reply": "I've created a ticket; a human colleague will contact you.",
                    "cited_sections": ["ESC-1"], "handover": {"issue": "repeat contact", "intent": "order_status",
                    "told_so_far": "called three times; verified with 1267", "checked": "order record", "reason": "ESC-1 rule 4"}}
        if "human" in newest.lower():
            return {"decision": "answer", "reply": "Sure, I can help with that.", "cited_sections": []}  # forces the guardrail
        return {"decision": "answer", "reply": intro + "Your order status is available.", "cited_sections": ["IDV-2"]}


def conversation(cid: str) -> list[dict]:
    with open(config.dataset_dir() / "practice" / "conversations.csv", encoding="utf-8", newline="") as f:
        return sorted((r for r in csv.DictReader(f) if r["conversation_id"] == cid), key=lambda r: int(r["turn"]))


KB = None


def kb():
    global KB
    if KB is None:
        KB = KnowledgeBase(use_dense="0")
    return KB


def test_id_extraction():
    assert extract_order_id("ऑर्डर नंबर है 7746 1439।") == "77461439"
    assert extract_order_id("order 89782837 please") == "89782837"
    assert extract_order_id("last four 4533") is None
    assert extract_verification("आख़िरी चार अंक १२६६ हैं।") == "1266"
    assert extract_verification("pincode 600042") == "600042"
    assert extract_verification("order 8978 2837 and last four 4533") == "4533"
    assert extract_verification("it was delivered on 20 July 2026") is None  # a year is not a PIN


def test_dates_match_dataset_answers():
    assert dates.add_business_days(date(2026, 10, 17), 7) == date(2026, 10, 27)  # PQ0022
    assert dates.add_business_days(date(2026, 10, 14), 5) == date(2026, 10, 21)  # PQ0028
    assert dates.add_business_days(date(2026, 10, 12), 3) == date(2026, 10, 15)  # PQ0027


def test_failed_verification_conversation():
    """PC021: unverified lookup -> wrong digits -> retry allowed -> second failure locks details."""
    agent = SupportAgent(llm=ScriptedLLM(), kb=kb(), mode="full")
    for r in conversation("PC021"):
        t = agent.respond(r["customer_message"])
        assert t.decision == r["expected_decision"], (r["turn"], t.decision, r["expected_decision"])
        assert scoring.tools_ok(r["expected_tool_calls"], t.tool_calls), (r["turn"], t.tool_calls)
    assert agent.state.locked and not agent.state.verified and agent.state.order is None


def test_repeat_contact_creates_real_ticket_without_leaking_digits():
    tools = config.load_tools()
    created = []
    real = tools.create_ticket
    tools.create_ticket = lambda **kw: (created.append(kw), real(**kw))[1]
    try:
        agent = SupportAgent(llm=ScriptedLLM(), kb=kb(), mode="full")
        for r in conversation("PC007"):
            t = agent.respond(r["customer_message"])
            assert t.decision == r["expected_decision"], (r["turn"], t.decision)
            assert scoring.tools_ok(r["expected_tool_calls"], t.tool_calls), (r["turn"], t.tool_calls)
        assert t.ticket_id and t.ticket_id.startswith("TKT")
        assert created and "1267" not in created[0]["summary"], "verification digits leaked into the ticket"
        assert all("1267" not in s for s in t.tool_call_strings()), "masked tool log must hide the secret"
    finally:
        tools.create_ticket = real
        (config.dataset_dir() / "tools" / "tickets_created.csv").unlink(missing_ok=True)


def test_never_claims_ticket_when_tool_fails():
    tools = config.load_tools()
    real = tools.create_ticket
    tools.create_ticket = lambda **kw: {"created": False, "error": "boom"}
    try:
        agent = SupportAgent(llm=ScriptedLLM(), kb=kb(), mode="full")
        t = agent.respond("I want a human please")
        assert t.decision == "escalate" and t.ticket_id is None
        assert "created a ticket" not in t.reply.lower() and "couldn't" in t.reply.lower()
    finally:
        tools.create_ticket = real


def test_rag_mode_has_no_tools_or_memory():
    agent = SupportAgent(llm=ScriptedLLM(), kb=kb(), mode="rag")
    t = agent.respond("order 7746 1439 status?")
    assert t.tool_calls == [] and agent.state.history == []


def test_scoring_rules():
    assert scoring.check_must_include("by the 27th of October 2026", "2026-10-27 / 27 October 2026")[0]
    assert scoring.check_must_include("five to seven business days", "5 ; 7")[0]
    assert not scoring.check_must_include("15 days", "5 ; 7")[0]
    assert not scoring.check_must_not("takes 5 to 7 days", "5 to 7 days ; 5-7 days")[0]


def test_retrieval_floor():
    """Raw-question retrieval (no LLM) must stay a usable baseline."""
    import pandas as pd
    q = pd.read_csv(config.dataset_dir() / "practice" / "questions.csv", dtype=str, keep_default_na=False)
    hits = []
    for _, r in q.iterrows():
        gold = [g for g in r.gold_evidence.split(" ; ") if g]
        if gold:
            ids = [h.chunk.id for h in kb().search([r.question], k=6)]
            hits.append(any(g in ids for g in gold))
    assert sum(hits) / len(hits) >= 0.88, sum(hits) / len(hits)


class ForgetfulLLM(ScriptedLLM):
    """Answers without ever introducing itself, and records the EVENTS block it was given."""
    events: list = []

    def chat_json(self, system, user, **kw):
        out = super().chat_json(system, user, **kw)
        if "understanding module" not in system:
            ForgetfulLLM.events.append(re.search(r"EVENTS THIS TURN:(.*?)\n\n=== DATE", user, re.S).group(1))
            out["reply"] = out["reply"].replace("I'm ZentroMart's AI support assistant. ", "")
        return out


def test_first_reply_always_says_it_is_an_ai():
    agent = SupportAgent(llm=ForgetfulLLM(), kb=kb(), mode="full")
    t1 = agent.respond("Hello, mujhe order ke baare mein help chahiye.")
    assert "AI support assistant" in t1.reply, t1.reply
    t2 = agent.respond("Order number hai 8978 2837.")
    assert "AI support assistant" not in t2.reply, "must not introduce itself twice"


def test_model_is_told_when_verification_just_succeeded_or_failed():
    ForgetfulLLM.events = []
    agent = SupportAgent(llm=ForgetfulLLM(), kb=kb(), mode="full")
    agent.respond("Order number hai 8978 2837.")
    agent.respond("Last four digits 0000 hain.")      # wrong
    agent.respond("Last four digits 4533 hain.")      # right (README example)
    ev = ForgetfulLLM.events
    assert "NOT verified yet" in ev[0]
    assert "DID NOT MATCH" in ev[1]
    assert "JUST SUCCEEDED" in ev[2]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
