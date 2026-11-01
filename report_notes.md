# Report notes (edit this file, then run: python make_report.py)

## Team

Team name: PiForge

Member: Priyanshu Raj Singh

GitHub: [https://github.com/Priyanshu-rajsingh/voiceops](https://github.com/Priyanshu-rajsingh/voiceops)

## What went wrong and what I would fix

- Free-tier LLM quotas ran out (Groq daily tokens and Gemini daily requests) before the full Step 1 vs Step 2 evaluation could finish. Therefore, the main report numbers are based on retrieval-only evaluation.

- On the full 52-question set, the prompt-only Step 1 baseline scored 17% overall using Groq (gpt-oss-120b). This result is based on an earlier terminal run.

- In a 5-question smoke test, Step 2 outperformed Step 1 (60% vs. 40%). Across 22 conversation turns, improvements to verification handling and enforcing the AI introduction increased passing turns from 36% to approximately 82–86%. These are small-sample results and should not be treated as a full evaluation.

- Speech-to-text and text-to-speech errors during voice testing could affect the overall user experience. I would further improve audio handling, transcription accuracy, and TTS reliability.

- Next, I would rerun the complete evaluation using a paid API key to obtain comparable full-dataset results and validate the improvements more reliably.

## Bonus work

- Implemented voice-based testing using Gnani speech-to-text and text-to-speech integration.
- Added multilingual voice testing with English and Hinglish sample recordings.
- Generated sample audio outputs and JSON/text responses for testing and debugging.
- Improved verification handling and AI introduction behavior in the VoiceOps agent.