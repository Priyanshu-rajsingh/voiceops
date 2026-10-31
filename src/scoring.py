"""Automatic scoring following DATA_DICTIONARY.md.

must_include   : facts separated by ' ; ' (all required); alternatives inside a fact separated by ' / '
                 (any one is enough). Ignore case; equivalent dates, spoken numbers and native digits accepted.
must_not_include: forbidden phrases separated by ' ; ' (a literal match is a violation; paraphrases cannot
                 be detected automatically - noted as a limitation in the report).
"""
from __future__ import annotations

import re

from . import dates

DEVA = str.maketrans("०१२३४५६७८९", "0123456789")
UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split())}
_NUMRUN = re.compile(
    r"\b(?:(?:" + "|".join(list(UNITS) + list(TENS) + ["hundred", "thousand", "lakh", "and"]) + r")\b[\s\-]*)+",
    re.I)


def _run_to_number(run: str) -> str | None:
    total = cur = 0
    seen = False
    for w in re.findall(r"[a-z]+", run.lower()):
        if w in UNITS:
            cur += UNITS[w]; seen = True
        elif w in TENS:
            cur += TENS[w]; seen = True
        elif w == "hundred":
            cur = (cur or 1) * 100; seen = True
        elif w == "thousand":
            total += (cur or 1) * 1000; cur = 0; seen = True
        elif w == "lakh":
            total += (cur or 1) * 100000; cur = 0; seen = True
    return str(total + cur) if seen else None


def words_to_digits(text: str) -> str:
    def sub(m: re.Match) -> str:
        n = _run_to_number(m.group(0))
        return f" {n} " if n is not None else m.group(0)

    return _NUMRUN.sub(sub, text)


def norm(text: str) -> str:
    t = text.translate(DEVA).lower()
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _match(alt: str, reply_n: str, reply_w: str, reply_dates: set) -> bool:
    d = set(dates.find_dates(alt))
    if d:
        return d <= reply_dates
    a = norm(alt)
    if not a:
        return True
    if re.fullmatch(r"[\d.]+", a):
        pat = rf"(?<!\d){re.escape(a)}(?!\d)"
        return bool(re.search(pat, reply_n) or re.search(pat, reply_w))
    return a in reply_n or a in reply_w


def check_must_include(reply: str, spec: str) -> tuple[bool, list[str]]:
    facts = [f.strip() for f in (spec or "").split(" ; ") if f.strip()]
    if not facts:
        return True, []
    reply_n = norm(reply)
    reply_w = norm(words_to_digits(reply_n))
    reply_dates = set(dates.find_dates(reply)) | set(dates.find_dates(reply_w))
    missing = []
    for fact in facts:
        alts = [a.strip() for a in fact.split(" / ") if a.strip()]
        if not any(_match(a, reply_n, reply_w, reply_dates) for a in alts):
            missing.append(fact)
    return (not missing), missing


def check_must_not(reply: str, spec: str) -> tuple[bool, list[str]]:
    bad_phrases = [p.strip() for p in (spec or "").split(" ; ") if p.strip()]
    reply_n = norm(reply)
    hits = [p for p in bad_phrases if norm(p) in reply_n]
    return (not hits), hits


def decision_ok(decision: str, expected: str, acceptable: str = "") -> bool:
    ok = {d.strip() for d in (acceptable or "").split(" ; ") if d.strip()} or {expected}
    return decision in ok


def retrieval_metrics(gold: str, retrieved: list[str], hard_negatives: str = "") -> dict | None:
    g = [x for x in (gold or "").split(" ; ") if x]
    if not g:
        return None
    ranks = [retrieved.index(x) for x in g if x in retrieved]
    hn = [x for x in (hard_negatives or "").split(" ; ") if x]
    hn_ranks = [retrieved.index(x) for x in hn if x in retrieved]
    return {"recall": sum(x in retrieved for x in g) / len(g), "hit": bool(ranks),
            "mrr": 1 / (1 + min(ranks)) if ranks else 0.0,
            "hard_neg_first": bool(hn_ranks) and (not ranks or min(hn_ranks) < min(ranks))}


_CALL = re.compile(r"(\w+)\(([^)]*)\)")


def parse_calls(spec: str) -> list[tuple[str, dict]]:
    out = []
    for name, args in _CALL.findall(spec or ""):
        d = {}
        for part in args.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                d[k.strip()] = v.strip()
        out.append((name, d))
    return out


def tools_ok(expected_spec: str, actual: list[dict]) -> bool:
    exp = parse_calls(expected_spec)
    for name, args in exp:
        if name == "create_ticket":
            if not any(c["name"] == "create_ticket" for c in actual):
                return False
        elif not any(c["name"] == name and str(c["args"].get("order_id")) == args.get("order_id")
                     and (str(c["args"].get("verification_value")) if "verification_value" in c["args"] else None)
                     == args.get("verification_value") for c in actual):
            return False
    if not any(n == "create_ticket" for n, _ in exp) and any(c["name"] == "create_ticket" for c in actual):
        return False  # a ticket nobody expected
    return True
