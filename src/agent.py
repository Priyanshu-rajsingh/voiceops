"""The ZentroMart support agent (Steps 1-3).

Modes
  baseline : Step 1 - system prompt only, no documents, no tools, no memory
  rag      : Step 2 - retrieval-grounded answers, stateless, no order tools
  full     : Step 3 - RAG + conversation memory + verified order lookup + tickets

Design: the LLM *understands* (router) and *writes* (answer); plain Python *decides what is allowed*:
identity verification (one retry), when tools run, date arithmetic, and whether a ticket was really created.
"""
from __future__ import annotations

import csv
import re
import threading
import time
from dataclasses import dataclass, field

from . import config, dates
from .kb import KnowledgeBase
from .llm import LLM, LLMError
from .prompts import ANSWER_SYSTEM, BASELINE_SYSTEM, ROUTER_SYSTEM

DECISIONS = {"answer", "clarify", "escalate", "refuse"}
ALWAYS_ON = ["ESC-1", "ESC-2", "ESC-4", "IDV-2", "IDV-3", "IDV-4", "KB-00.1"]  # short rules, always in context
_TICKET_LOCK = threading.Lock()
_DEVA = str.maketrans("०१२३४५६७८९", "0123456789")
ORDER_RE = re.compile(r"(?<!\d)(\d{4})[\s\-]?(\d{4})(?!\d)")
CLAIM_TICKET_RE = re.compile(r"(created|raised|logged|opened|filed|बना (?:दिया|दी)|दर्ज|bana diya|bana di)", re.I)
ORDER_SKIP = {"customer_id", "delivery_pincode", "verified", "found", "items", "verification_required"}

AI_RE = re.compile(r"\bAI\b|artificial|एआई|ए\.?\s?आई|कृत्रिम", re.I)
INTRO = {"en": "I'm ZentroMart's AI support assistant. ",
         "hi": "मैं ZentroMart की AI सहायक हूँ। ",
         "hi-en": "Main ZentroMart ki AI support assistant hoon. "}

FALLBACKS = {
    "trouble": {"en": "Sorry, I'm having trouble right now. Could you please try again in a moment?",
                "hi": "क्षमा करें, अभी मुझे दिक्कत आ रही है। कृपया थोड़ी देर बाद फिर कोशिश करें।",
                "hi-en": "Sorry, abhi mujhe thodi dikkat aa rahi hai. Kripya thodi der baad phir try karein."},
    "ticket_failed": {"en": "I'm sorry, I couldn't pass this to a colleague just now. Please try again in a moment or contact support from your registered phone number.",
                      "hi": "क्षमा करें, अभी मैं इसे सहयोगी तक नहीं पहुँचा पाई। कृपया थोड़ी देर बाद कोशिश करें या अपने पंजीकृत फ़ोन नंबर से सपोर्ट से संपर्क करें।",
                      "hi-en": "Sorry, abhi main ise colleague tak nahi pahuncha paayi. Kripya thodi der baad try karein ya registered phone number se support se contact karein."},
    "human": {"en": " A human colleague will contact you about this.",
              "hi": " हमारे एक मानव सहयोगी इस बारे में आपसे संपर्क करेंगे।",
              "hi-en": " Hamare ek human colleague is baare mein aapse contact karenge."},
}


def _fb(kind: str, lang: str) -> str:
    return FALLBACKS[kind].get(lang, FALLBACKS[kind]["en"])


def normalize_digits(text: str) -> str:
    return text.translate(_DEVA).replace("\u200b", "")


def extract_order_id(text: str) -> str | None:
    m = ORDER_RE.search(normalize_digits(text))
    return m.group(1) + m.group(2) if m else None


def _strip_dates(text: str) -> str:
    text = dates._DATE_ISO.sub(" ", text)
    return dates._DATE_TEXT.sub(" ", text)


def extract_verification(text: str) -> str | None:
    t = _strip_dates(ORDER_RE.sub(" ", normalize_digits(text)))
    m = re.search(r"(?<!\d)(\d{6}|\d{4})(?!\d)", t)
    return m.group(1) if m else None


def group4(oid: str) -> str:
    return f"{oid[:4]} {oid[4:]}" if len(oid) == 8 else oid


def detect_language(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    low = text.lower()
    if re.search(r"\b(mujhe|mera|meri|hai|hain|kya|kab|nahi|kaise|karna|chahiye|aap|ko|ka|ki|se|wapas|paise|bhej)\b", low):
        return "hi-en"
    return "en"


@dataclass
class Route:
    language: str = "en"
    english_query: str = ""
    search_terms: list[str] = field(default_factory=list)
    order_id: str | None = None
    verification_value: str | None = None
    dates: list[str] = field(default_factory=list)
    states_issue: bool = False
    wants_human: bool = False
    legal_threat: bool = False
    safety_risk: bool = False
    unauthorised_activity: bool = False
    issue_summary: str = ""

    @property
    def must_escalate(self) -> bool:
        return self.wants_human or self.legal_threat or self.safety_risk or self.unauthorised_activity


@dataclass
class Turn:
    user: str
    reply: str = ""
    decision: str = "answer"
    language: str = "en"
    retrieved: list[str] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    support: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)  # raw (in memory only)
    ticket_id: str | None = None
    query: str = ""
    latency: float = 0.0
    error: str | None = None

    @property
    def chunks_used(self) -> list[str]:
        return list(dict.fromkeys(self.retrieved + self.support + self.cited))

    def tool_call_strings(self, mask: bool = True) -> list[str]:
        out = []
        for c in self.tool_calls:
            args = dict(c["args"])
            if mask and "verification_value" in args:
                args["verification_value"] = "****"
            out.append(f"{c['name']}({', '.join(f'{k}={v}' for k, v in args.items())})")
        return out

    def to_dict(self) -> dict:
        return {"user": self.user, "reply": self.reply, "decision": self.decision, "language": self.language,
                "retrieved": self.retrieved, "support_chunks": self.support, "cited_sections": self.cited,
                "chunks_used": self.chunks_used, "tool_calls": self.tool_call_strings(mask=True),
                "ticket_id": self.ticket_id, "search_query": self.query, "latency_s": round(self.latency, 2),
                "error": self.error}


@dataclass
class Session:
    history: list[tuple[str, str]] = field(default_factory=list)
    order_id: str | None = None
    verified: bool = False
    attempts: int = 0
    locked: bool = False
    order: dict | None = None
    replies: int = 0
    secrets: set[str] = field(default_factory=set)
    last_query: str = ""


class SupportAgent:
    def __init__(self, llm: LLM | None = None, kb: KnowledgeBase | None = None, mode: str = "full",
                 top_k: int | None = None):
        assert mode in ("baseline", "rag", "full")
        self.mode = mode
        self.llm = llm or LLM()
        self.kb = kb if kb is not None else (KnowledgeBase() if mode != "baseline" else None)
        self.top_k = top_k or config.get_settings().top_k
        self.tools = config.load_tools() if mode == "full" else None
        self.products = self._load_products() if mode != "baseline" else {}
        self.state = Session()
        self._events: list[str] = []

    # ------------------------------------------------------------------ public API
    def reset(self) -> None:
        self.state = Session()

    def respond(self, text: str, channel: str = "chat") -> Turn:
        t0 = time.time()
        turn = Turn(user=text)
        try:
            if self.mode == "baseline":
                self._baseline(turn)
            else:
                self._grounded(turn, channel)
        except LLMError as e:
            turn.error = str(e)
            turn.language = detect_language(text)
            turn.reply, turn.decision = _fb("trouble", turn.language), "clarify"
        turn.latency = time.time() - t0
        if self.mode == "full":
            self.state.history += [("customer", text), ("assistant", turn.reply)]
            self.state.history = self.state.history[-10:]
            self.state.replies += 1
        return turn

    # ------------------------------------------------------------------ Step 1
    def _baseline(self, turn: Turn) -> None:
        data = self.llm.chat_json(BASELINE_SYSTEM, turn.user, max_tokens=500)
        turn.reply = str(data.get("reply", "")).strip()
        turn.decision = data.get("decision") if data.get("decision") in DECISIONS else "answer"
        turn.language = detect_language(turn.user)

    # ------------------------------------------------------------------ Steps 2-3
    def _grounded(self, turn: Turn, channel: str) -> None:
        st = self.state
        memory = self.mode == "full"
        self._events = []
        route = self._route(turn.user)
        turn.language = route.language

        if self.mode == "full":
            self._identity_step(turn, route)

        queries = [f"{route.english_query} {' '.join(route.search_terms)}".strip(), turn.user]
        if memory and st.last_query:  # follow-ups like "and how long will that take?" keep the earlier topic
            queries.append(st.last_query)
        hits = self.kb.search(queries, k=self.top_k)
        turn.query = queries[0]
        if memory:
            st.last_query = route.english_query
        forced = []
        if route.must_escalate:
            forced += ["ESC-1"] + (["DWM-5"] if route.safety_risk else [])
        have = {h.chunk.id for h in hits}
        for cid in forced:
            if cid not in have and self.kb.get(cid):
                from .kb import Hit
                hits.append(Hit(self.kb.get(cid), 0.0, support=True))
        turn.retrieved = [h.chunk.id for h in hits if not h.support]
        turn.support = [h.chunk.id for h in hits if h.support]

        ctx = self._context(turn, route, hits, channel, memory)
        data = self.llm.chat_json(ANSWER_SYSTEM, ctx, max_tokens=900)
        decision = str(data.get("decision", "answer")).lower().strip()
        decision = decision if decision in DECISIONS else "answer"
        reply = str(data.get("reply", "")).strip() or _fb("trouble", route.language)
        lang = str(data.get("reply_language") or route.language)
        turn.language = lang
        turn.cited = [c for c in (data.get("cited_sections") or []) if isinstance(c, str) and self.kb.get(c.strip())]
        turn.cited = [c.strip() for c in turn.cited]

        # --- guardrails -------------------------------------------------------------------------
        if st.locked and not st.verified:
            reply = self._redact(reply)
        wants_ticket = decision == "escalate" or route.must_escalate or (
            self.mode == "full" and CLAIM_TICKET_RE.search(reply) and "ticket" in reply.lower())
        if wants_ticket:
            if decision != "escalate":
                if not re.search(r"human|colleague|सहयोगी|team", reply, re.I):
                    reply += _fb("human", lang)
                decision = "escalate"
            if self.mode == "full":
                ok = self._create_ticket(turn, route, data.get("handover") or {})
                if not ok:
                    reply = _fb("ticket_failed", lang)
        reply = reply.replace("{ticket_id}", "").strip()
        if st.replies == 0 and not AI_RE.search(reply):  # STY-1: the first reply must say it is an AI assistant
            reply = INTRO.get(lang, INTRO["en"]) + reply
        turn.reply, turn.decision = reply, decision

    # ------------------------------------------------------------------ helpers: understanding
    def _route(self, text: str) -> Route:
        st = self.state
        hist = "\n".join(f"{who}: {msg}" for who, msg in st.history[-6:]) if self.mode == "full" else ""
        prompt = (f"Conversation so far:\n{hist or '(none)'}\n\nNEWEST customer message:\n{text}\n\n"
                  f"Today is {config.TODAY.isoformat()}.")
        try:
            d = self.llm.chat_json(ROUTER_SYSTEM, prompt, max_tokens=500)
        except LLMError:
            d = {}
        r = Route(
            language=str(d.get("language") or detect_language(text)),
            english_query=str(d.get("english_query") or text),
            search_terms=[str(x) for x in (d.get("search_terms") or [])][:8],
            order_id=str(d["order_id"]) if d.get("order_id") else None,
            verification_value=str(d["verification_value"]) if d.get("verification_value") else None,
            dates=[str(x) for x in (d.get("dates") or [])],
            states_issue=bool(d.get("states_issue")), wants_human=bool(d.get("wants_human")),
            legal_threat=bool(d.get("legal_threat")), safety_risk=bool(d.get("safety_risk")),
            unauthorised_activity=bool(d.get("unauthorised_activity")),
            issue_summary=str(d.get("issue_summary") or ""),
        )
        # regex wins over the model for IDs (exact, deterministic); the model fills the gaps (spoken digits)
        r.order_id = extract_order_id(text) or (re.sub(r"\D", "", r.order_id) if r.order_id else None)
        if r.order_id and len(r.order_id) != 8:
            r.order_id = None
        return r

    # ------------------------------------------------------------------ helpers: identity + tools
    def _identity_step(self, turn: Turn, route: Route) -> None:
        st = self.state
        if route.order_id and route.order_id != st.order_id:  # a new order: start verification afresh
            st.order_id, st.verified, st.attempts, st.locked, st.order = route.order_id, False, 0, False, None
            self._call(turn, "lookup_order", {"order_id": route.order_id})
            self._events.append(f"A new order ID ({group4(route.order_id)}) was received this turn and is NOT verified yet. Ask for ONE proof: the last four digits of the registered phone number OR the delivery pincode. Decision \"clarify\". Share no order details.")
        if st.order_id and not st.verified and not st.locked:
            proof = extract_verification(turn.user) or (
                re.sub(r"\D", "", route.verification_value or "") if route.verification_value else None)
            if proof and len(proof) in (4, 6):
                res = self._call(turn, "lookup_order", {"order_id": st.order_id, "verification_value": proof})
                st.secrets.add(proof)
                if res.get("found") and res.get("verified"):
                    st.verified, st.order = True, res
                    self._events.append("VERIFICATION JUST SUCCEEDED in this message. Say in one short clause that the customer is verified (never repeat their digits), then help using the order record; if they have not yet said what they need, ask how you can help with this order. Decision \"answer\". Do NOT ask for any further proof.")
                else:
                    st.attempts += 1
                    st.locked = st.attempts >= 2
                    kind = "last four phone digits" if len(proof) == 4 else "delivery pincode"
                    other = "delivery pincode" if len(proof) == 4 else "last four phone digits"
                    if st.locked:
                        self._events.append(f"The {kind} did NOT match and this was the SECOND failed attempt. Share no order details; say the details do not match the registered ones, suggest contacting support from the registered phone number, and offer general policy help. Decision \"refuse\".")
                    else:
                        self._events.append(f"The {kind} the customer just gave DID NOT MATCH (attempt 1 of 2). Say it did not match and ask for the {other} instead. Never reveal the correct value. Decision \"clarify\".")

    def _call(self, turn: Turn, name: str, args: dict) -> dict:
        res = getattr(self.tools, name)(**args)
        turn.tool_calls.append({"name": name, "args": args})
        return res

    def _create_ticket(self, turn: Turn, route: Route, h: dict) -> bool:
        st = self.state
        summary = (
            f"Issue: {h.get('issue') or route.issue_summary or turn.user}. "
            f"Intent: {h.get('intent') or 'support'}. "
            f"Order ID: {st.order_id or 'not provided'} (identity {'verified' if st.verified else 'not verified'}). "
            f"Customer said: {h.get('told_so_far') or turn.user}. "
            f"Checked: {h.get('checked') or 'policy documents' + (' and the verified order record' if st.verified else '')}. "
            f"Escalation reason: {h.get('reason') or 'a human must decide under the escalation policy'}."
        )
        summary = self._redact(summary)
        with _TICKET_LOCK:
            res = self.tools.create_ticket(summary=summary, order_id=st.order_id,
                                           reason=self._redact(str(h.get("reason") or "")) or None,
                                           intent=str(h.get("intent") or "") or None)
        turn.tool_calls.append({"name": "create_ticket", "args": {}})
        if res.get("created"):
            turn.ticket_id = res.get("ticket_id")
            return True
        return False

    def _redact(self, text: str) -> str:
        for s in self.state.secrets:
            text = text.replace(s, "[redacted]")
        return text

    # ------------------------------------------------------------------ helpers: context building
    def _load_products(self) -> dict:
        with open(config.dataset_dir() / "data" / "products.csv", encoding="utf-8", newline="") as f:
            return {r["sku"]: r for r in csv.DictReader(f)}

    def _order_facts(self, order: dict) -> str:
        lines = [f"{k}: {v}" for k, v in order.items() if k not in ORDER_SKIP and v not in (None, "")]
        for it in order.get("items", []):
            p = self.products.get(it["sku"], {})
            lines.append(
                f"item line {it['line_no']}: sku {it['sku']} | {it['title']} | qty {it['qty']} | unit price "
                f"₹{it['unit_price_inr']} | return_category {it['return_category']} | fulfilled_by "
                f"{p.get('fulfilled_by', '?')} | assured {p.get('assured', '?')} | warranty_months "
                f"{p.get('warranty_months', '?')}")
        return "\n".join(lines)

    def _hints(self, order: dict) -> list[str]:
        out = []
        c = order.get("contacts_last_7_days") or 0
        if c >= 2:
            out.append(f"contacts_last_7_days = {c}: if the customer is now raising an issue about this order it is "
                       "their THIRD or later contact within 7 days -> escalate (ESC-1 rule 4).")
        sku = order.get("damage_sku")
        if sku:
            for it in order.get("items", []):
                if it["sku"] == sku:
                    value = int(it["qty"]) * int(it["unit_price_inr"])
                    if value > 10000:
                        out.append(f"damage report value = ₹{value}, above ₹10,000 -> Priority Resolution; "
                                   "escalate, do not approve (DWM-3, ESC-1 rule 1).")
        return out

    def _context(self, turn: Turn, route: Route, hits, channel: str, memory: bool) -> str:
        st = self.state
        key_dates = [d for d in (dates.parse_date(x) for x in route.dates) if d] + dates.find_dates(turn.user)
        if st.verified and st.order:
            for k, v in st.order.items():
                if k.endswith(("_on", "_date", "_at")) or k in ("placed_at", "promised_date"):
                    d = dates.parse_date(v)
                    if d:
                        key_dates.append(d)
        if self.mode != "full":
            verification = "ORDER LOOKUP IS NOT AVAILABLE in this mode. Do not discuss any specific order."
        elif st.verified:
            verification = "VERIFIED - you may use the order record below."
        elif st.locked:
            verification = "FAILED TWICE - LOCKED. Share NO order details."
        elif st.order_id and st.attempts == 1:
            verification = (f"FAILED ONCE for order {group4(st.order_id)} - ask again for the last four phone digits "
                            "or the delivery pincode (one retry left).")
        elif st.order_id:
            verification = (f"PENDING - order ID {group4(st.order_id)} received; ask for the last four digits of the "
                            "registered phone number OR the delivery pincode.")
        else:
            verification = "NOT STARTED - no order ID yet (ask for it only if the question needs an order)."
        tickets = ("A ticket will be created by the system when you choose \"escalate\"."
                   if self.mode == "full" else "Tickets are not available in this mode; say a human colleague can help.")
        guides = [self.kb.get(c) for c in ALWAYS_ON if self.kb.get(c)]
        parts = [
            f"CHANNEL: {channel}",
            f"FIRST REPLY: {'yes' if st.replies == 0 else 'no'}",
            f"CUSTOMER LANGUAGE: {route.language}",
            f"VERIFICATION: {verification}",
            f"TICKETS: {tickets}",
            ("EVENTS THIS TURN:\n- " + "\n- ".join(self._events)) if self._events else "EVENTS THIS TURN: none",
            "ESCALATION SIGNALS: " + (", ".join(n for n, f in [
                ("customer asked for a human", route.wants_human), ("legal action mentioned", route.legal_threat),
                ("product safety risk", route.safety_risk), ("activity customer did not make",
                                                              route.unauthorised_activity)] if f) or "none from the message"),
            "\n=== DATE HELPER ===\n" + dates.cheat_sheet(key_dates),
            "\n=== RETRIEVED DOCUMENTS (most relevant first) ===\n" + "\n\n".join(h.chunk.text for h in hits),
            "\n=== ALWAYS-ON RULES ===\n" + "\n\n".join(c.text for c in guides),
        ]
        if st.verified and st.order:
            parts.append("\n=== VERIFIED ORDER RECORD ===\n" + self._order_facts(st.order))
            parts += [f"SYSTEM CHECK: {h}" for h in self._hints(st.order)]
        if memory and st.history:
            parts.append("\n=== CONVERSATION SO FAR ===\n" + "\n".join(f"{w}: {m}" for w, m in st.history[-8:]))
        parts.append(f"\n=== NEWEST CUSTOMER MESSAGE ===\n{turn.user}")
        return "\n".join(parts)
