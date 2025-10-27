# Bangla Enablement TODO

- [x] Audit dependencies and environment variables to reconcile `medgemma/requirements.txt#L1` with the Bangla `requirements.txt` (add `google-cloud-aiplatform`, `ujson`, pin versions) and document required ENV keys from `README_BANGLA.md`.
- [x] Replace the diskcache layer (`medgemma/cache.py#L1`) with the Bangla `PersistentCache` pattern, including text/audio subdirectories, stats tracking, and UTF-8 writes.
- [x] Rebuild Gemini patient simulation in Bangla by porting logic from `gemini_bangla.py` into `medgemma/gemini.py#L1`, including history-aware prompts, Bangla personas, and cache hooks.
- [x] Reimplement MedGemma calling code in `medgemma/medgemma.py#L1` to initialize Vertex AI via service-account JSON, inject Bangla system prompts, add language hints, and cache responses.
- [x] Swap the TTS pipeline: adapt `gemini_tts_bangla.py` into `medgemma/gemini_tts.py#L1`, emitting Bangla (`bn-BD`) audio, caching base64 payloads, and honoring the `GENERATE_SPEECH` flag.
- [x] Overhaul the interview flow in `medgemma/interview_simulator.py#L1` to the Bangla conversation model (MedGemma questions, Gemini patient replies, completion detection, Bangla report/transcript generation).
- [x] Redesign the Flask surface in `medgemma/app.py#L1` to the REST endpoints used by the Bangla version (start interview, send message, generate report, get transcript, cache stats) and wire in the new cache plus session handling.
- [x] Update supporting assets: align docs (`medgemma/README.md#L1`) with Bangla behavior, review `frontend/` expectations (patient data vs. new flow), and stage basic regression checks for caching/TTS toggles.
