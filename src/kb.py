"""Knowledge base: chunking, indexing and hybrid retrieval.

* Every Markdown section (## [SECTION-ID] ...) is one chunk; the section ID, doc status,
  doc type and effective_from date travel with it. Catalog rows become chunks too
  (id = products.csv:SKU).
* Retrieval = Reciprocal-Rank-Fusion of
    - BM25 over word tokens,
    - TF-IDF vectors over character n-grams (robust to Hinglish spelling variants),
    - optional dense multilingual embeddings (sentence-transformers, if installed).
  A small precedence prior (current policy > agent guide > help > FAQ > marketing;
  superseded documents down-weighted) is applied after fusion (KB-00.1).
"""
from __future__ import annotations

import csv
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import config

DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
HEADER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
SECTION_RE = re.compile(r"^##\s*\[([^\]]+)\]\s*(.*)$", re.M)
SKU_RE = re.compile(r"\b([A-Z]{3}-\d{4})\b", re.I)

PRIOR = {"policy": 1.0, "agent_guide": 0.95, "meta": 0.9, "help": 0.9, "faq": 0.8, "marketing": 0.65, "catalog": 1.0}
SUPERSEDED_PRIOR = 0.7

# tiny Hinglish/Hindi -> English policy vocabulary bridge for the sparse retrievers
LEXICON = {
    "wapas": "return", "vapas": "return", "वापस": "return", "लौटा": "return", "lautana": "return",
    "paise": "refund money", "paisa": "refund money", "पैसे": "refund money", "पैसा": "refund money",
    "रिफंड": "refund", "रिफ़ंड": "refund", "ऑर्डर": "order", "कैंसिल": "cancel", "रद्द": "cancel",
    "डिलीवरी": "delivery", "डिलिवरी": "delivery", "गारंटी": "warranty", "वारंटी": "warranty",
    "kab": "when timeline", "kitne": "how many", "din": "days", "दिन": "days",
    "kharab": "damaged defective", "toot": "damaged broken", "टूट": "damaged broken", "खराब": "damaged defective",
    "pickup": "reverse pickup", "पिकअप": "reverse pickup", "bhej": "send return", "bhejna": "send return",
    "samay": "time", "समय": "time", "kitna": "how much", "shulk": "fee", "charge": "fee",
    "madad": "help", "insaan": "human agent", "इंसान": "human agent", "बात": "talk",
}


def tokenize(text: str) -> list[str]:
    text = text.lower().translate(DEVA_DIGITS)
    toks, cur = [], []
    for ch in text:
        if ch.isalnum() or unicodedata.category(ch).startswith("M"):
            cur.append(ch)
        elif cur:
            toks.append("".join(cur))
            cur = []
    if cur:
        toks.append("".join(cur))
    out: list[str] = []
    for t in toks:
        if t in LEXICON:
            out += [_stem(w) for w in LEXICON[t].split()]
        out.append(_stem(t))
    return out


def _stem(w: str) -> str:
    if not w.isascii() or len(w) <= 3:
        return w
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


@dataclass(eq=False)
class Chunk:
    id: str
    text: str
    doc_id: str
    title: str
    doc_type: str
    status: str
    effective_from: str
    language: str
    source: str

    @property
    def index_text(self) -> str:
        return f"{self.title} {self.id} {self.text}"

    def brief(self) -> dict:
        return {"id": self.id, "doc_id": self.doc_id, "doc_type": self.doc_type, "status": self.status,
                "title": self.title}


@dataclass
class Hit:
    chunk: Chunk
    score: float
    support: bool = False  # added as interpretation context (precedence / business-day rules)


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = self.len.mean() if len(docs) else 1.0
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def _parse_header(text: str) -> tuple[dict, str]:
    m = HEADER_RE.match(text)
    meta: dict[str, str] = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        text = text[m.end():]
    return meta, text


def load_chunks(root) -> list[Chunk]:
    chunks: list[Chunk] = []
    for md in sorted((root / "kb").glob("*.md")):
        meta, body = _parse_header(md.read_text(encoding="utf-8"))
        matches = list(SECTION_RE.finditer(body))
        preface = body[: matches[0].start()] if matches else body
        note = " ".join(l.lstrip("> ").strip() for l in preface.splitlines()
                        if l.strip() and not l.startswith("# "))
        status = meta.get("status", "current")
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
            sec = body[m.end():end].strip()
            tag = f"status: {status}"
            if status == "superseded":
                tag += f", SUPERSEDED by {meta.get('superseded_by', 'a newer document')}"
            head = (f"[{m.group(1)}] {m.group(2).strip()} (document: {meta.get('title', md.stem)}; "
                    f"type: {meta.get('doc_type', '?')}; {tag}; effective from {meta.get('effective_from', '?')})")
            text = head + ("\nDocument note: " + note if status == "superseded" and note else "") + "\n" + sec
            chunks.append(Chunk(m.group(1), text, meta.get("doc_id", md.stem), m.group(2).strip(),
                                meta.get("doc_type", "help"), status, meta.get("effective_from", ""),
                                meta.get("language", "en"), md.name))
    with open(root / "data" / "products.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            cid = f"products.csv:{r['sku']}"
            text = (f"[{cid}] Product catalog row - title: {r['title']}; brand: {r['brand']}; "
                    f"return_category: {r['return_category']}; price_inr: {r['price_inr']}; "
                    f"fulfilled_by: {r['fulfilled_by']}; assured: {r['assured']}; "
                    f"warranty_months: {r['warranty_months']}; description: {r['description']}")
            chunks.append(Chunk(cid, text, "products.csv", r["title"], "catalog", "current", "", "en", "products.csv"))
    return chunks


class KnowledgeBase:
    def __init__(self, root=None, use_dense: str | None = None, embed_model: str | None = None):
        self.root = config.dataset_dir(root)
        self.chunks = load_chunks(self.root)
        self.by_id = {c.id: c for c in self.chunks}
        texts = [c.index_text for c in self.chunks]
        self._bm25 = BM25([tokenize(t) for t in texts])
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
        self._tfidf_mat = self._tfidf.fit_transform([t.translate(DEVA_DIGITS) for t in texts])
        self._dense = None
        st = config.get_settings()
        mode = (use_dense or st.use_dense)
        if mode != "0":
            self._init_dense(embed_model or st.embed_model, forced=(mode == "1"))
        self.retrievers = ["bm25", "tfidf"] + (["dense"] if self._dense else [])

    # -- dense (optional) ---------------------------------------------------
    def _init_dense(self, model_name: str, forced: bool):
        try:
            import importlib.util

            if not forced and importlib.util.find_spec("sentence_transformers") is None:
                return
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            pre = "passage: " if "e5" in model_name.lower() else ""
            emb = model.encode([pre + c.index_text for c in self.chunks], normalize_embeddings=True, batch_size=16)
            self._dense = (model, np.asarray(emb), "query: " if "e5" in model_name.lower() else "")
        except Exception as e:  # network/model problems must never break the agent
            print(f"[kb] dense embeddings unavailable ({e.__class__.__name__}); using BM25 + TF-IDF only")

    # -- search ---------------------------------------------------------------
    def _rankings(self, query: str) -> dict[str, np.ndarray]:
        out = {"bm25": self._bm25.scores(tokenize(query))}
        qv = self._tfidf.transform([query.translate(DEVA_DIGITS)])
        out["tfidf"] = (self._tfidf_mat @ qv.T).toarray().ravel()
        if self._dense:
            model, emb, pre = self._dense
            out["dense"] = emb @ np.asarray(model.encode([pre + query], normalize_embeddings=True))[0]
        return out

    def _fuse(self, queries: list[tuple[str, float]], allowed: np.ndarray, prior: bool = True) -> np.ndarray:
        weights = {"bm25": 1.0, "tfidf": 0.8, "dense": 1.2}
        fused = np.zeros(len(self.chunks))
        for q, qw in queries:
            for name, scores in self._rankings(q).items():
                scores = np.where(allowed, scores, 0.0)
                if not scores.any():
                    continue
                for rank, idx in enumerate(np.argsort(-scores)[:30]):
                    if scores[idx] > 0:
                        fused[idx] += qw * weights[name] / (60 + rank)
        if prior:
            for i, c in enumerate(self.chunks):
                fused[i] *= SUPERSEDED_PRIOR if c.status == "superseded" else PRIOR.get(c.doc_type, 0.9)
        return fused

    def _catalog_matches(self, queries: list[str]) -> list[Chunk]:
        """Products the customer is talking about: exact SKU, exact title, or a very close fuzzy title."""
        joined = " ".join(queries)
        low = joined.lower()
        found: list[Chunk] = []
        for m in SKU_RE.findall(joined):
            c = self.by_id.get(f"products.csv:{m.upper()}")
            if c:
                found.append(c)
        for c in self.chunks:
            if c.doc_type == "catalog" and len(c.title) > 6 and c.title.lower() in low:
                found.append(c)
        if not found:  # fuzzy: speech-to-text may garble product names
            qv = self._tfidf.transform([joined.translate(DEVA_DIGITS)])
            sims = (self._tfidf_mat @ qv.T).toarray().ravel()
            cat_idx = [i for i, c in enumerate(self.chunks) if c.doc_type == "catalog"]
            ranked = sorted(cat_idx, key=lambda i: -sims[i])
            best = ranked[0]
            if sims[best] >= self.FUZZY_PRODUCT_MIN:
                found += [self.chunks[i] for i in ranked[:2] if sims[i] >= 0.85 * sims[best]]
        return list(dict.fromkeys(found))[:2]

    FUZZY_PRODUCT_MIN = 0.30

    def search(self, queries: list[str], k: int = 6, support: bool = True) -> list[Hit]:
        """Return up to k policy/guide chunks (+ matched catalog rows + interpretation context)."""
        queries = [q for q in queries if q and q.strip()]
        is_policy = np.array([c.doc_type != "catalog" for c in self.chunks])
        products = self._catalog_matches(queries)
        weighted = [(q, 1.0 if i == 0 else 0.7) for i, q in enumerate(queries)]
        for p in products:  # the product's return category code leads to the right policy rows (RET-2 table)
            m = re.search(r"return_category: (\w+)", p.text)
            if m:
                weighted.append((f"return window category {m.group(1)} policy", 0.6))
        if products:  # sections that explain the catalog columns the row contains (warranty, Assured, seller, ...)
            weighted.append(("catalog fields assured warranty_months warranty claim fulfilled_by seller marketplace policy", 0.5))
        fused = self._fuse(weighted, is_policy)
        order = [i for i in np.argsort(-fused) if fused[i] > 0][:k]
        hits = [Hit(self.chunks[i], float(fused[i])) for i in order]
        top = max((h.score for h in hits), default=1.0)
        have = {h.chunk.id for h in hits}
        for p in products:
            hits.insert(0, Hit(p, top * 1.05))
            have.add(p.id)
        if support:
            texts = " ".join(h.chunk.text.lower() for h in hits)
            raw = self._fuse(weighted, is_policy, prior=False)
            overlap = [self.chunks[i] for i in np.argsort(-raw)[:8] if raw[i] > 0]
            noisy = any(c.doc_type in ("faq", "help", "marketing") or c.status == "superseded" for c in overlap) \
                or any(h.chunk.doc_type in ("faq", "help", "marketing") or h.chunk.status == "superseded"
                       for h in hits)
            wanted = []
            if noisy:
                wanted.append("KB-00.1")  # which document wins
            if "business day" in texts:
                wanted.append("KB-00.3")  # how to count business days
            for cid in wanted:
                if cid in self.by_id and cid not in have:
                    hits.append(Hit(self.by_id[cid], 0.0, support=True))
        return hits

    def get(self, cid: str) -> Chunk | None:
        return self.by_id.get(cid)
