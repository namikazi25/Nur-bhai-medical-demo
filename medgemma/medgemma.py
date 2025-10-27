# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# MedGemma Bangla integration using Vertex AI
import json
import os
from typing import Any, Dict, List, Optional

from google.cloud import aiplatform
from google.oauth2 import service_account

from cache import PersistentCache, cache as default_cache

# Optional HF local backend
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    _TRANSFORMERS_AVAILABLE = False
try:
    import torch
except Exception:
    torch = None  # type: ignore

# Backend selection
BACKEND = os.environ.get("MEDGEMMA_BACKEND", "vertex").lower()
FALLBACK = os.environ.get("MEDGEMMA_FALLBACK", "").lower()
HF_MODEL_ID = os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
HF_DEVICE_MAP = os.environ.get("MEDGEMMA_DEVICE_MAP", "auto")
HF_DTYPE = os.environ.get("MEDGEMMA_DTYPE", "bfloat16")

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
GCP_MEDGEMMA_ENDPOINT = os.environ.get("GCP_MEDGEMMA_ENDPOINT")
GCP_SERVICE_ACCOUNT_KEY = os.environ.get("GCP_MEDGEMMA_SERVICE_ACCOUNT_KEY")

_vertex_initialized = False
_vertex_credentials: Optional[service_account.Credentials] = None
_hf_model = None
_hf_processor = None


def _initialize_vertex_ai() -> None:
    global _vertex_initialized, _vertex_credentials
    if _vertex_initialized:
        return

    if GCP_SERVICE_ACCOUNT_KEY:
        try:
            credentials_info = json.loads(GCP_SERVICE_ACCOUNT_KEY)
            _vertex_credentials = service_account.Credentials.from_service_account_info(credentials_info)
            aiplatform.init(project=GCP_PROJECT, location=GCP_LOCATION, credentials=_vertex_credentials)
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Vertex AI initialization failed: {exc}")
            _vertex_credentials = None
            aiplatform.init(project=GCP_PROJECT, location=GCP_LOCATION)
    else:
        aiplatform.init(project=GCP_PROJECT, location=GCP_LOCATION)

    _vertex_initialized = True


def _build_bangla_system_prompt() -> str:
    return """আপনি একজন সহায়ক চিকিৎসা সহকারী। আপনার নির্দেশাবলী:
- বাংলায় স্পষ্ট ও পেশাদার ভঙ্গিতে উত্তর দিন
- রোগীর উদ্বেগ বোঝার জন্য সহানুভূতিশীলভাবে প্রশ্ন করুন
- উপসর্গের সময়কাল, তীব্রতা, উদ্দীপক এবং সংশ্লিষ্ট লক্ষণ নিয়ে তথ্য সংগ্রহ করুন
- প্রয়োজনে চিকিৎসা পরিভাষার ব্যাখ্যা দিন
- কোনও মেডিকেল সিদ্ধান্ত বা পরামর্শ দিবেন না
"""


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _build_messages(
    prompt: str,
    conversation_history: List[Dict[str, str]],
    extra_system_prompts: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": _build_bangla_system_prompt()}]

    for extra_prompt in extra_system_prompts or []:
        messages.append({"role": "system", "content": extra_prompt})

    for message in conversation_history:
        messages.append({"role": message["role"], "content": message["content"]})

    if prompt:
        messages.append({"role": "user", "content": prompt})

    return messages


def generate_medgemma_response(
    prompt: str,
    conversation_history: List[Dict[str, str]],
    cache: Optional[PersistentCache] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_prompts: Optional[List[str]] = None,
) -> str:
    """
    Generate a Bangla response from MedGemma with caching.
    """

    cache = cache or default_cache
    # Local HF backend path
    if BACKEND == "local":
        text = _generate_with_hf(prompt, conversation_history, system_prompts or [], temperature, max_tokens)
        if text:
            return text
        # If HF failed, optionally fall back to Vertex based on env knob
        if FALLBACK != "vertex":
            return "স্থানীয় মডেল লোড করা যায়নি। MEDGEMMA_FALLBACK=vertex সেট করলে Vertex AI ব্যবহৃত হবে।"
        if not GCP_MEDGEMMA_ENDPOINT:
            return "স্থানীয় মডেল লোড করা যায়নি এবং MedGemma endpoint কনফিগার নেই।"

    serialized_context = {
        "prompt": prompt,
        "history": conversation_history,
        "system_prompts": system_prompts or [],
    }

    conversation_context = json.dumps(
        serialized_context,
        ensure_ascii=False,
    )

    cached_response = cache.get(prompt, context=conversation_context)
    if cached_response:
        return cached_response

    # Vertex backend path
    if not GCP_MEDGEMMA_ENDPOINT:
        return "MedGemma endpoint is not configured. অনুগ্রহ করে GCP_MEDGEMMA_ENDPOINT সেট করুন।"
    _initialize_vertex_ai()
    try:
        endpoint = aiplatform.Endpoint(GCP_MEDGEMMA_ENDPOINT)
        messages = _build_messages(prompt, conversation_history, extra_system_prompts=system_prompts)
        response = endpoint.predict(
            instances=[
                {
                    "messages": messages,
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "language": "bn",
                }
            ]
        )
        predictions = getattr(response, "predictions", None)
        if predictions:
            generated_text = predictions[0].get("content", "")
            if generated_text:
                cache.set(prompt, generated_text, context=conversation_context)
                return generated_text
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"MedGemma API error: {exc}")

    return "দুঃখিত, একটি ত্রুটি ঘটেছে। অনুগ্রহ করে আবার চেষ্টা করুন।"


def _hf_resolve_dtype():
    if torch is None:
        return None
    requested = HF_DTYPE.lower()
    mapping = {
        "bfloat16": getattr(torch, "bfloat16", None),
        "float16": getattr(torch, "float16", None),
        "float32": getattr(torch, "float32", None),
    }
    dtype = mapping.get(requested)

    if dtype is torch.bfloat16 and torch.cuda.is_available():
        # Older GPUs (e.g., T4 with compute capability < 8.0) do not support bfloat16 well.
        try:
            major = torch.cuda.get_device_properties(0).major
            if major < 8:
                return mapping.get("float16")
        except Exception:
            return mapping.get("float16")

    return dtype


def _initialize_hf_local():
    global _hf_model, _hf_processor
    if _hf_model is not None and _hf_processor is not None:
        return True
    if not _TRANSFORMERS_AVAILABLE:
        return False
    try:
        dtype = _hf_resolve_dtype()
        _hf_model = AutoModelForImageTextToText.from_pretrained(
            HF_MODEL_ID,
            torch_dtype=dtype,
            device_map=HF_DEVICE_MAP,
        )
        _hf_processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
        return True
    except Exception as exc:
        print(f"HF model load failed: {exc}")
        _hf_model = None
        _hf_processor = None
        return False


def _build_hf_messages(
    prompt: str,
    conversation_history: List[Dict[str, str]],
    system_prompts: List[str],
) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []

    messages.append({"role": "system", "content": [{"type": "text", "text": _build_bangla_system_prompt()}]})

    for extra in system_prompts:
        messages.append({"role": "system", "content": [{"type": "text", "text": extra}]})

    history_lines: List[str] = []
    for msg in conversation_history:
        speaker = "সহায়ক" if msg.get("role") == "assistant" else "রোগী"
        history_lines.append(f"{speaker}: {msg.get('content', '')}")

    combined_prompt = prompt or ""
    if history_lines:
        history_text = "\n".join(history_lines)
        if combined_prompt:
            combined_prompt = (
                f"পূর্ববর্তী কথোপকথন:\n{history_text}\n\nবর্তমান নির্দেশ:\n{combined_prompt}"
            )
        else:
            combined_prompt = f"পূর্ববর্তী কথোপকথন:\n{history_text}"

    if combined_prompt:
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": combined_prompt}],
        })

    return messages


def _generate_with_hf(
    prompt: str,
    conversation_history: List[Dict[str, str]],
    system_prompts: List[str],
    temperature: float,
    max_tokens: int,
) -> Optional[str]:
    # Use cached response first (shared cache keying)
    conv_context = json.dumps(
        {"prompt": prompt, "history": conversation_history, "system_prompts": system_prompts}, ensure_ascii=False
    )
    cached = default_cache.get(prompt, context=conv_context)
    if cached:
        return cached

    if not _initialize_hf_local():
        return None
    try:
        messages_hf = _build_hf_messages(prompt, conversation_history, system_prompts)
        inputs = _hf_processor.apply_chat_template(
            messages_hf,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        if torch is not None:
            inputs = inputs.to(_hf_model.device, dtype=getattr(torch, HF_DTYPE, None))
            input_len = inputs["input_ids"].shape[-1]
            do_sample = temperature is not None and temperature > 0.0
            with torch.inference_mode():
                generation = _hf_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                )
                generation = generation[0][input_len:]
            decoded = _hf_processor.decode(generation, skip_special_tokens=True)
        else:
            # Fallback: attempt CPU run without dtype/acceleration
            input_len = inputs["input_ids"].shape[-1]
            generation = _hf_model.generate(**inputs, max_new_tokens=max_tokens)
            generation = generation[0][input_len:]
            decoded = _hf_processor.decode(generation, skip_special_tokens=True)

        text = (decoded or "").strip()
        if text:
            default_cache.set(prompt, text, context=conv_context)
            return text
        return None
    except Exception as exc:
        print(f"HF generation failed: {exc}")
        return None


def generate_medical_summary(
    patient_info: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    cache: Optional[PersistentCache] = None,
) -> str:
    """
    Create a Bangla medical summary from conversation history.
    """

    conversation_text = "\n".join(
        f"{'সহায়ক' if msg['role'] == 'assistant' else 'রোগী'}: {msg['content']}"
        for msg in conversation_history
    )

    summary_prompt = f"""নিম্নলিখিত রোগীর তথ্য এবং কথোপকথনের উপর ভিত্তি করে বাংলায় একটি বিস্তারিত প্রাক-পরিদর্শন সারাংশ তৈরি করুন।

রোগীর তথ্য:
- নাম: {patient_info.get('name')}
- বয়স: {patient_info.get('age')}
- লিঙ্গ: {patient_info.get('gender')}

কথোপকথন:
{conversation_text}

সারাংশে নিম্নলিখিত বিভাগ অন্তর্ভুক্ত করুন:
১. প্রধান অভিযোগ
২. বর্তমান অসুস্থতার ইতিহাস
৩. সংশ্লিষ্ট উপসর্গ
৪. প্রাসঙ্গিক চিকিৎসা ইতিহাস
৫. সম্ভাব্য ডিফারেনশিয়াল ডায়াগনসিস
৬. প্রস্তাবিত পরীক্ষা এবং মূল্যায়ন
"""

    return generate_medgemma_response(
        prompt=summary_prompt,
        conversation_history=[],
        cache=cache,
        temperature=0.5,
        max_tokens=2048,
    )


def evaluate_interview_quality(
    conversation_history: List[Dict[str, str]],
    reference_diagnosis: str,
    cache: Optional[PersistentCache] = None,
) -> str:
    """
    Evaluate interview quality in Bangla with MedGemma.
    """

    conversation_text = "\n".join(
        f"{'সহায়ক' if msg['role'] == 'assistant' else 'রোগী'}: {msg['content']}"
        for msg in conversation_history
    )

    eval_prompt = f"""নিম্নলিখিত চিকিৎসা সাক্ষাৎকার মূল্যায়ন করুন।

রেফারেন্স ডায়াগনসিস: {reference_diagnosis}

কথোপকথন:
{conversation_text}

নিম্নলিখিত মানদণ্ডের উপর ভিত্তি করে একটি মূল্যায়ন রিপোর্ট প্রদান করুন:
১. তথ্য সংগ্রহের সম্পূর্ণতা
২. প্রশ্নের প্রাসঙ্গিকতা এবং মান
৩. যোগাযোগ দক্ষতা
৪. সঠিক ডায়াগনসিসের দিকে পথ
৫. উন্নতির ক্ষেত্র

মূল্যায়ন বাংলায় গঠনমূলক ও বিস্তারিত করুন।"""

    return generate_medgemma_response(
        prompt=eval_prompt,
        conversation_history=[],
        cache=cache,
        temperature=0.3,
        max_tokens=2048,
    )


def medgemma_get_text_response(messages: List[Dict[str, Any]], **kwargs: Any) -> str:
    """
    Backwards-compatible wrapper that accepts OpenAI-style messages.
    """

    cache = kwargs.get("cache")
    temperature = kwargs.get("temperature", 0.7)
    max_tokens = kwargs.get("max_tokens", 1024)

    system_prompts: List[str] = []
    conversation_history: List[Dict[str, str]] = []

    for entry in messages:
        role = entry.get("role")
        content = _normalize_content(entry.get("content", ""))
        if role == "system":
            system_prompts.append(content)
        elif role in {"assistant", "user"}:
            conversation_history.append({"role": role, "content": content})

    final_prompt = ""
    if conversation_history and conversation_history[-1]["role"] == "user":
        final_prompt = conversation_history.pop()["content"]

    return generate_medgemma_response(
        prompt=final_prompt,
        conversation_history=conversation_history,
        cache=cache,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompts=system_prompts,
    )
