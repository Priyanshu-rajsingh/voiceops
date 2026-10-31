#!/usr/bin/env python3
"""Demo interface (for the 2-minute video): speak or type, the agent answers in text and voice.

    pip install -r requirements-extra.txt
    python app.py            # opens http://127.0.0.1:7860
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from src.agent import SupportAgent
from src.kb import KnowledgeBase
from src.llm import LLM
from src.voice import Voice, speech_text

LANGS = {"English / Hinglish (en-IN)": "en-IN", "Hindi (hi-IN)": "hi-IN", "Tamil (ta-IN)": "ta-IN",
         "Telugu (te-IN)": "te-IN", "Kannada (kn-IN)": "kn-IN", "Marathi (mr-IN)": "mr-IN",
         "Bengali (bn-IN)": "bn-IN", "Gujarati (gu-IN)": "gu-IN", "Malayalam (ml-IN)": "ml-IN",
         "Punjabi (pa-IN)": "pa-IN"}

agent = SupportAgent(llm=LLM(), kb=KnowledgeBase(), mode="full")
voice = Voice()
OUT = Path(tempfile.mkdtemp(prefix="voiceops_"))
N = {"i": 0}


def turn(audio, typed, lang_label, log):
    log = log or ""
    if audio:
        heard = voice.transcribe(audio, LANGS[lang_label]).text
    else:
        heard = (typed or "").strip()
    if not heard:
        reply, wav, src = "Sorry, I didn't catch that. Could you please say it again?", None, ""
        log += f"\n**Customer:** _(nothing recognised)_\n\n**Agent:** {reply}\n"
    else:
        t = agent.respond(heard, channel="voice")
        reply = t.reply
        N["i"] += 1
        wav = str(OUT / f"reply_{N['i']}.wav")
        wav = wav if voice.speak(speech_text(reply), t.language, wav) else None
        calls = ", ".join(t.tool_call_strings()) or "none"
        src = (f"**Decision:** {t.decision}  \n**Tools:** {calls}  \n**Sources:** {', '.join(t.chunks_used) or 'none'}")
        log += f"\n**Customer:** {heard}\n\n**Agent:** {reply}\n"
    return log, wav, src, None, ""


def reset():
    agent.reset()
    return "", None, "", None, ""


with gr.Blocks(title="ZentroMart VoiceOps") as demo:
    gr.Markdown("## ZentroMart support - talk to the agent\nSpeak into the microphone (or type), then press **Send**.")
    lang = gr.Dropdown(list(LANGS), value="English / Hinglish (en-IN)", label="Speaking language")
    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak")
        txt = gr.Textbox(label="...or type", lines=2)
    with gr.Row():
        send = gr.Button("Send", variant="primary")
        clear = gr.Button("New customer")
    chat = gr.Markdown(label="Conversation")
    out_audio = gr.Audio(label="Agent voice", autoplay=True, type="filepath")
    sources = gr.Markdown()
    send.click(turn, [mic, txt, lang, chat], [chat, out_audio, sources, mic, txt])
    clear.click(reset, None, [chat, out_audio, sources, mic, txt])

if __name__ == "__main__":
    demo.launch()
