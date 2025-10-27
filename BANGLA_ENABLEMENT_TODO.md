# Bangla Enablement TODO

- [x] Audit dependencies and environment variables to reconcile `medgemma/requirements.txt#L1` with the Bangla `requirements.txt` (add `google-cloud-aiplatform`, `ujson`, pin versions) and document required ENV keys from `README_BANGLA.md`.
- [x] Replace the diskcache layer (`medgemma/cache.py#L1`) with the Bangla `PersistentCache` pattern, including text/audio subdirectories, stats tracking, and UTF-8 writes.
- [x] Rebuild Gemini patient simulation in Bangla by porting logic from `gemini_bangla.py` into `medgemma/gemini.py#L1`, including history-aware prompts, Bangla personas, and cache hooks.
- [x] Reimplement MedGemma calling code in `medgemma/medgemma.py#L1` to initialize Vertex AI via service-account JSON, inject Bangla system prompts, add language hints, and cache responses.
- [x] Swap the TTS pipeline: adapt `gemini_tts_bangla.py` into `medgemma/gemini_tts.py#L1`, emitting Bangla (`bn-BD`) audio, caching base64 payloads, and honoring the `GENERATE_SPEECH` flag.
- [x] Overhaul the interview flow in `medgemma/interview_simulator.py#L1` to the Bangla conversation model (MedGemma questions, Gemini patient replies, completion detection, Bangla report/transcript generation).
- [x] Redesign the Flask surface in `medgemma/app.py#L1` to the REST endpoints used by the Bangla version (start interview, send message, generate report, get transcript, cache stats) and wire in the new cache plus session handling.
- [x] Update supporting assets: align docs (`medgemma/README.md#L1`) with Bangla behavior, review `frontend/` expectations (patient data vs. new flow), and stage basic regression checks for caching/TTS toggles.

## Local Deployment (Hugging Face) TODO

- [x] Add local backend switch in `medgemma/medgemma.py`:
  - Env flag `MEDGEMMA_BACKEND=local` to bypass Vertex and use HF.
  - Env `MEDGEMMA_MODEL_ID` defaulting to `google/medgemma-4b-it` (dev-friendly).
  - Lazy-load `AutoProcessor` and `AutoModelForImageTextToText` and reuse across calls.
  - Accept OpenAI-style messages, apply chat template, generate, decode; integrate `PersistentCache`.

- [x] Dependencies for local run:
  - Add `transformers>=4.50.0`, `accelerate` to `medgemma/requirements.txt` (do not pin torch here; document install via PyTorch site).
  - Optional: `bitsandbytes` for 4-bit quantization; guard load if available.

- [x] Env and config knobs (.env / README):
  - `MEDGEMMA_BACKEND=local | vertex`
  - `MEDGEMMA_MODEL_ID=google/medgemma-4b-it` (or `google/medgemma-27b-text-it`)
  - `MEDGEMMA_DEVICE_MAP=auto` (pass to HF `from_pretrained`)
  - `MEDGEMMA_DTYPE=bfloat16` (or `float16`, `int4` if quantized)
  - `GENERATE_SPEECH=false` (fully local, no TTS network)
  - Document `huggingface-cli login` and gated model access requirement.

- [x] Local backend implementation details:
  - Build messages: include Bangla system + history + user final turn.
  - Use `processor.apply_chat_template(..., add_generation_prompt=True)` and `model.generate(...)`.
  - Respect `temperature` if feasible; otherwise deterministic decode for MVP.
  - Cache results using existing `PersistentCache` keying on prompt+history.
  - Add graceful fallback: if local load/generate fails and `MEDGEMMA_FALLBACK=vertex`, route to Vertex.

- [x] Documentation updates:
  - Add “Local (HF) run” section to `medgemma/README.md` with setup, GPU notes, env, and curl tests.
  - Provide `.env`/`env.list` examples for local backend.
  - Note VRAM requirements (4B vs 27B) and quantization options.

- [x] Optional GPU containerization (advanced):
  - Add `Dockerfile.gpu` based on `nvidia/cuda` with PyTorch+Transformers.
  - Add `run_local_gpu.sh` using `--gpus all` and local backend env.

- [ ] Smoke/regression tests for local mode:
  - Start interview, send message, generate report via curl.
  - Repeat calls to verify cache hits increase.
  - Toggle `GENERATE_SPEECH` off/on and confirm audio presence/absence.
  - Monitor memory and latency; document expected ranges.
