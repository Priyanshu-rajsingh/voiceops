"""Business-day and calendar helpers (the LLM must not do date arithmetic itself).

Rules (KB-00.3): a business day is Monday-Saturday excluding the holidays in
data/holidays.csv. Refund/logistics timelines count business days from the day
the refund is initiated (day 0). Return windows count calendar days.
"""
from __future__ import annotations

import csv
import re
from datetime import date, datetime, timedelta

from . import config

_HOLIDAYS: dict[date, str] | None = None

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
HI_MONTHS = ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त",
             "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
_ORD = "st|nd|rd|th"
_MONTH_ALT = "|".join(MONTHS + [m[:3] for m in MONTHS] + HI_MONTHS + ["फरवरी", "सितम्बर", "अक्तूबर"])
_DATE_TEXT = re.compile(
    rf"(?P<d>\d{{1,2}})(?:{_ORD})?\s*(?:of\s+)?(?P<m>{_MONTH_ALT})\.?,?\s*(?P<y>\d{{4}})?", re.I)
_DATE_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def holidays() -> dict[date, str]:
    global _HOLIDAYS
    if _HOLIDAYS is None:
        _HOLIDAYS = {}
        with open(config.dataset_dir() / "data" / "holidays.csv", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                _HOLIDAYS[date.fromisoformat(row["date"])] = row["name"]
    return _HOLIDAYS


def is_business_day(d: date) -> bool:
    return d.weekday() != 6 and d not in holidays()


def add_business_days(d: date, n: int) -> date:
    step = 1 if n >= 0 else -1
    left = abs(n)
    while left:
        d += timedelta(days=step)
        if is_business_day(d):
            left -= 1
    return d


def business_days_between(a: date, b: date) -> int:
    """Number of business days after `a` up to and including `b` (negative if b < a)."""
    if b < a:
        return -business_days_between(b, a)
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        n += is_business_day(d)
    return n


def month_index(name: str) -> int | None:
    name = name.lower().strip(".")
    for i, m in enumerate(MONTHS):
        if name == m or (len(name) == 3 and m.startswith(name)):
            return i + 1
    for i, m in enumerate(HI_MONTHS):
        if name == m:
            return i + 1
    return {"फरवरी": 2, "सितम्बर": 9, "अक्तूबर": 10}.get(name)


def find_dates(text: str, default_year: int | None = None) -> list[date]:
    """Find dates written as 2026-10-17, '17 October 2026', '17th Oct', Hindi months."""
    text = text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    default_year = default_year or config.TODAY.year
    found: list[date] = []
    for m in _DATE_ISO.finditer(text):
        try:
            found.append(date(int(m[1]), int(m[2]), int(m[3])))
        except ValueError:
            pass
    for m in _DATE_TEXT.finditer(text):
        mi = month_index(m["m"])
        if not mi:
            continue
        try:
            found.append(date(int(m["y"]) if m["y"] else default_year, mi, int(m["d"])))
        except ValueError:
            pass
    return list(dict.fromkeys(found))


def parse_date(value) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)[:19]).date()
    except ValueError:
        return None


def fmt(d: date) -> str:
    return f"{d.isoformat()} ({d.strftime('%A')})"


def spoken(d: date) -> str:
    """'24th of October 2026' - used in spoken replies (STY-3)."""
    n = d.day
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix} of {d.strftime('%B')} {d.year}"


def cheat_sheet(key_dates: list[date], today: date | None = None) -> str:
    """Pre-computed date facts so the model never has to count days itself."""
    today = today or config.TODAY
    hol = ", ".join(f"{d.isoformat()} {n}" for d, n in sorted(holidays().items()))
    lines = [
        f"TODAY = {fmt(today)}.",
        f"Holidays (not business days; Sundays also excluded): {hol}.",
        "Business days after TODAY: " + ", ".join(
            f"+{i}={add_business_days(today, i).isoformat()}" for i in (1, 2, 3, 4, 5, 7, 10)) + ".",
    ]
    for d in sorted(set(key_dates)):
        if d == today:
            continue
        delta = (today - d).days
        when = f"{delta} calendar days before TODAY" if delta > 0 else f"{-delta} calendar days after TODAY"
        bd = business_days_between(d, today)
        lines.append(
            f"{fmt(d)} is {when}; business days from it to TODAY: {bd}. "
            "Business days after it: " + ", ".join(
                f"+{i}={add_business_days(d, i).isoformat()}" for i in (1, 2, 3, 4, 5, 6, 7, 10)) + ". "
            "Calendar days after it: " + ", ".join(
                f"+{i}={(d + timedelta(days=i)).isoformat()}" for i in (2, 7, 10, 15, 30)) + ".")
    return "\n".join(lines)
