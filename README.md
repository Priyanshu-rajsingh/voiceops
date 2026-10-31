# VoiceOps 2026 - ZentroMart voice support agent

A customer-support agent for the fictional store **ZentroMart**. It answers from the company documents (RAG),
holds a multi-turn conversation (memory, identity check, tools, human handover) and talks on the phone using
**Gnani STT and TTS** (the only speech services used).

**Public repository:** https://github.com/<your-username>/<your-repo>

```
customer audio -> Gnani STT -> router (LLM) -> identity gate + tools -> hybrid retriever -> answer (LLM) -> guardrails -> Gnani TTS -> spoken reply
```

## Setup

Python 3.10+. `ffmpeg` is optional (used to split clips over 55 s and to convert unusual audio formats).

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then fill in GNANI_API_KEY, LLM_PROVIDER, LLM_MODEL, LLM_API_KEY
```

**Dataset.** Not included (per the brief). Extract `Capstone_Project_Dataset_FTTT_2026.zip` next to this folder (or
inside it), or set `DATASET_DIR` / `DATASET_ZIP` in `.env`. The code also finds the original ZIP by itself.
`python Capstone_Project_Dataset_FTTT_2026/validate_dataset.py` should print PASS.

## Run command (batch, for the organizers)

```bash
python run.py --audio_dir ./test_audio --out_dir ./outputs
```

For every audio file (wav, mp3, m4a, ogg, flac, aac, opus, webm) it writes to `--out_dir`:

| file | content |
| --- | --- |
| `<name>.json` | what STT heard, reply text, decision, **chunks_used** (retrieved + cited section IDs), tool calls, ticket id, timings, errors |
| `<name>.txt` | the reply as text |
| `<name>_reply.wav` | the reply spoken with Gnani TTS |
| `results.csv` | one-line summary per file |

Every file is independent (fresh memory). Files named like `PC003_t1.wav, PC003_t2.wav` are treated as consecutive
turns of one conversation. One failing file never stops the batch: it still gets a spoken "please say it again".
Verification digits are masked (`****`) in everything that is saved.

## Evaluate (Step 1 vs Step 2 vs Step 3)

```bash
python -m src.evaluate --retrieval-only        # no keys needed: retrieval quality of the raw question
python -m src.evaluate                         # Step 1, Step 2, Step 3 (single questions) and the 23 conversations
python make_report.py                          # builds report.pdf from those numbers
```

Add `--asr` to use the speech-to-text style question text, `--limit 10` for a quick smoke test.
Results: `eval_out/summary.json` and per-row CSVs.

## Demo app and tests

```bash
pip install -r requirements-extra.txt && python app.py     # microphone demo in the browser
python -m tests.test_offline && python -m tests.test_run   # no keys needed
```

## How it works

* **Step 1 - prompt only** (`prompts.BASELINE_SYSTEM`): role, limits, AI disclosure; no documents.
* **Step 2 - RAG** (`src/kb.py`): one chunk per `## [SECTION-ID]` with status / type / effective date; catalog rows as
  chunks; BM25 + character-n-gram TF-IDF (+ dense multilingual embeddings if `requirements-extra.txt` is installed)
  fused with reciprocal-rank fusion; precedence prior (policy > guide > help > FAQ > marketing, superseded
  down-weighted); an LLM rewrites the customer's words into an English policy query first; products pull in
  their return category; `KB-00.1` / `KB-00.3` are added when sources overlap or business days matter.
* **Step 3 - conversation** (`src/agent.py`): session memory; deterministic identity gate (order ID -> unverified
  lookup; one value at a time; one retry; lock after the second failure); precomputed date helper
  (`src/dates.py`, business days skip Sundays + holidays); escalation rules create a ticket and the reply claims it only if
  `created=True`; verification digits are redacted from tickets and logs.
* **Step 4 - voice** (`src/voice.py`): Gnani STT (`en-IN` accepts English mixed with Hindi, `hi-IN` fallback,
  `format="transcribe"` so digits arrive as digits) and Gnani TTS (voice/language chosen per reply; text prepared for the ear).

## Limitations

The identity check is the dataset's synthetic mock, not real authentication. Evaluation uses string matching on
52 questions / 23 conversations, so it can mis-score paraphrases. Dense embeddings are optional; without them
retrieval relies on the LLM query rewrite plus sparse search.
