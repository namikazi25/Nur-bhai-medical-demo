import json
import os
from typing import Dict, List, Optional

import google.generativeai as genai

from cache import PersistentCache, cache as default_cache

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def generate_gemini_response(
    prompt: str,
    message: str,
    conversation_history: List[Dict[str, str]],
    cache: Optional[PersistentCache] = None,
    temperature: float = 0.8,
) -> str:
    """
    Generate a Bangla patient response using Gemini with conversation-aware caching.

    Args:
        prompt: Patient persona/context prompt in Bangla.
        message: Latest assistant message to respond to.
        conversation_history: Ordered history of conversation dictionaries with keys
            "role" (assistant|user) and "content".
        cache: Optional PersistentCache instance; defaults to module cache.
        temperature: Sampling temperature for Gemini generation.
    """

    cache = cache or default_cache

    conversation_context = json.dumps(
        {"prompt": prompt, "message": message, "history": conversation_history},
        ensure_ascii=False,
    )
    cached_response = cache.get(message, context=conversation_context)
    if cached_response:
        return cached_response

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        full_prompt = f"""{prompt}

পূর্ববর্তী কথোপকথন:
"""
        for history in conversation_history[-6:]:
            speaker = "সহায়ক" if history.get("role") == "assistant" else "রোগী"
            full_prompt += f"{speaker}: {history.get('content', '')}\n"

        full_prompt += f"\nসহায়ক: {message}\n\nরোগী হিসেবে বাংলায় প্রাকৃতিকভাবে সংক্ষিপ্ত উত্তর দিন (২-৩ বাক্য):"

        response = model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=256,
                candidate_count=1,
            ),
        )

        patient_response = (response.text or "").strip()
        if patient_response:
            cache.set(message, patient_response, context=conversation_context)
            return patient_response
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Gemini API error: {exc}")

    return "দুঃখিত, আমি এই মুহূর্তে উত্তর দিতে পারছি না।"


def create_patient_context(patient_data: Dict) -> str:
    """
    Create a Bangla persona prompt for the simulated patient.
    """

    context = f"""আপনি {patient_data['name']} নামে একজন রোগী। আপনার তথ্য:

ব্যক্তিগত তথ্য:
- বয়স: {patient_data['age']} বছর
- লিঙ্গ: {patient_data['gender']}

স্বাস্থ্য অবস্থা:
- বর্তমান সমস্যা: {patient_data.get('chief_complaint', 'সাধারণ স্বাস্থ্য পরীক্ষা')}
"""

    if patient_data.get("symptoms"):
        context += "\nউপসর্গ:\n"
        for symptom in patient_data["symptoms"]:
            context += f"- {symptom}\n"

    if patient_data.get("medical_history"):
        context += f"\nচিকিৎসা ইতিহাস: {patient_data['medical_history']}\n"

    if patient_data.get("medications"):
        context += "\nবর্তমান ওষুধ:\n"
        for medication in patient_data["medications"]:
            context += f"- {medication}\n"

    context += """\n
আপনার আচরণ:
- স্বাভাবিক রোগীর মতো কথা বলুন, চিকিৎসা বিশেষজ্ঞের মতো নয়
- সংক্ষিপ্ত এবং সরল উত্তর দিন
- আবেগ প্রকাশ করুন (উদ্বেগ, ব্যথা, ভয় ইত্যাদি)
- সব প্রশ্নের উত্তর না জানলে "আমি নিশ্চিত নই" বলুন
- শুধুমাত্র জিজ্ঞাসিত প্রশ্নের উত্তর দিন
- অতিরিক্ত বিস্তারিত তথ্য স্বেচ্ছায় দেবেন না যদি না জিজ্ঞাসা করা হয়

বাংলায় উত্তর দিন।"""

    return context


def generate_patient_scenarios() -> List[Dict]:
    """
    Provide sample Bangla patient scenarios used to seed simulations.
    """

    return [
        {
            "name": "রহিম উদ্দিন",
            "age": 45,
            "gender": "পুরুষ",
            "chief_complaint": "বুকে ব্যথা এবং শ্বাসকষ্ট",
            "symptoms": [
                "বুকের বাম পাশে চাপ অনুভব",
                "হাঁটলে শ্বাসকষ্ট হয়",
                "মাঝে মাঝে ঘাম হয়",
                "সমস্যা ৩ দিন ধরে",
            ],
            "medical_history": "উচ্চ রক্তচাপ, ডায়াবেটিস",
            "medications": ["মেটফরমিন ৫০০ মিগ্রা", "অ্যামলোডিপিন ৫ মিগ্রা"],
        },
        {
            "name": "সালমা বেগম",
            "age": 32,
            "gender": "মহিলা",
            "chief_complaint": "মাথাব্যথা এবং জ্বর",
            "symptoms": [
                "তীব্র মাথাব্যথা ২ দিন ধরে",
                "জ্বর ১০১-১০২ ডিগ্রি",
                "শরীর ব্যথা",
                "ক্ষুধা কম লাগে",
            ],
            "medical_history": "কোন দীর্ঘমেয়াদী রোগ নেই",
            "medications": ["প্যারাসিটামল প্রয়োজন অনুযায়ী"],
        },
        {
            "name": "করিম মিয়া",
            "age": 58,
            "gender": "পুরুষ",
            "chief_complaint": "হাঁটুতে ব্যথা",
            "symptoms": [
                "হাঁটু শক্ত লাগে সকালে",
                "সিঁড়ি উঠতে কষ্ট হয়",
                "কখনো কখনো ফুলে যায়",
                "সমস্যা ৬ মাস ধরে ধীরে ধীরে বাড়ছে",
            ],
            "medical_history": "আর্থ্রাইটিসের পারিবারিক ইতিহাস",
            "medications": ["আইবুপ্রোফেন প্রয়োজন অনুযায়ী"],
        },
    ]


def gemini_get_text_response(prompt: str, **kwargs) -> str:
    """
    Backwards-compatible helper that treats the prompt as the final message.
    """

    return generate_gemini_response(
        prompt="",
        message=prompt,
        conversation_history=[],
        cache=kwargs.get("cache"),
        temperature=kwargs.get("temperature", 0.8),
    )

