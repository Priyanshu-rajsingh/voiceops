"""Batch-run and LLM-client tests with local stand-ins (no keys, no network): python -m tests.test_run"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run as runmod
from src import config
from src.llm import LLM, LLMError
from tests.test_offline import ScriptedLLM


class FakeVoice:
    def __init__(self, *a, **k):
        self.said = {}

    def transcribe(self, path, language=None):
        from src.voice import STTResult
        said = {"q_blank": "", "q_boom": "BOOM"}.get(Path(path).stem)
        text = {"PC021_t1": "नमस्ते, मुझे अपने ऑर्डर के बारे में मदद चाहिए।", "PC021_t2": "ऑर्डर नंबर है 7746 1439।",
                "PC021_t3": "आख़िरी चार अंक 1266 हैं।"}.get(Path(path).stem, said if said is not None else "hello")
        if text == "BOOM":
            raise RuntimeError("stt exploded")
        return STTResult(text, "en-IN")

    def speak(self, text, lang, out):
        Path(out).write_bytes(b"RIFFfake")
        self.said[str(out)] = text
        return True


def test_batch_run_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        audio, out = Path(td) / "audio", Path(td) / "out"
        audio.mkdir()
        for n in ["q_hello", "q_blank", "q_boom", "PC021_t1", "PC021_t2", "PC021_t3"]:
            (audio / f"{n}.wav").write_bytes(b"x")
        with mock.patch.object(runmod, "LLM", ScriptedLLM), mock.patch.object(runmod, "Voice", FakeVoice):
            assert runmod.main(["--audio_dir", str(audio), "--out_dir", str(out)]) == 0
        results = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in out.glob("*.json")}
        assert set(results) == {"q_hello", "q_blank", "q_boom", "PC021_t1", "PC021_t2", "PC021_t3"}
        for stem in results:  # text, audio and JSON for every single file
            assert (out / f"{stem}.txt").exists() and (out / f"{stem}_reply.wav").exists()
        assert (out / "results.csv").exists()
        assert results["q_blank"]["decision"] == "clarify" and "say it again" in results["q_blank"]["reply_text"]
        assert results["q_boom"]["error"] and results["q_boom"]["audio_file"]  # failed STT still gets a spoken reply
        # turns of one conversation share memory: the order ID from t2 is used when digits arrive in t3
        assert results["PC021_t2"]["tool_calls"] == ["lookup_order(order_id=77461439)"]
        assert results["PC021_t3"]["tool_calls"] == ["lookup_order(order_id=77461439, verification_value=****)"]
        assert "1266" not in json.dumps(results["PC021_t3"]), "verification digits must not be saved"
        assert "AI support assistant" in results["q_hello"]["reply_text"]


class Mock(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        Mock.seen.append((self.path, dict(self.headers), body))
        if self.path.endswith("/messages"):
            out = {"content": [{"type": "text", "text": '```json\n{"decision": "answer", "reply": "hi"}\n```'}]}
        elif "max_tokens" in body and self.path.startswith("/strict"):
            self.send_response(400); self.end_headers()
            self.wfile.write(b'{"error": "Unsupported parameter: max_tokens, use max_completion_tokens"}'); return
        else:
            out = {"choices": [{"message": {"content": '{"decision": "clarify", "reply": "ok"}'}}]}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_llm_client_both_api_styles():
    srv = HTTPServer(("127.0.0.1", 0), Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    a = LLM("anthropic", "m", "k", base)
    assert a.chat_json("sys", "usr")["reply"] == "hi"
    path, headers, body = Mock.seen[-1]
    assert path == "/v1/messages" and headers["x-api-key"] == "k" and body["system"] == "sys"
    o = LLM("openai", "m", "k", base)
    assert o.chat_json("sys", "usr")["decision"] == "clarify"
    path, headers, body = Mock.seen[-1]
    assert path == "/chat/completions" and headers["authorization"] == "Bearer k"
    assert body["messages"][0] == {"role": "system", "content": "sys"} and body["response_format"]
    # providers that reject max_tokens get one automatic retry with max_completion_tokens
    s = LLM("openai", "m", "k", base + "/strict")
    assert s.chat_json("sys", "usr")["reply"] == "ok"
    assert "max_completion_tokens" in Mock.seen[-1][2]
    srv.shutdown()


def test_missing_key_is_a_clear_error():
    try:
        LLM("openai", "m", "", "http://x")
    except LLMError as e:
        assert "API key" in str(e)
    else:
        raise AssertionError("expected LLMError")


if __name__ == "__main__":
    failed = 0
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        try:
            fn(); print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
