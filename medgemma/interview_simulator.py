from typing import Any, Dict, List

from cache import PersistentCache
from gemini import create_patient_context, generate_gemini_response
from gemini_tts import generate_speech
from medgemma import generate_medgemma_response, generate_medical_summary


class InterviewSimulator:
    """Simulate a medical interview with Bangla language support."""

    COMPLETION_MARKER = "তথ্য সংগ্রহ সম্পন্ন"
    MAX_TURNS = 14  # 7 exchanges (assistant+patient)

    def __init__(self, patient_data: Dict[str, Any], cache: PersistentCache):
        self.patient_data = patient_data
        self.cache = cache
        self.conversation_history: List[Dict[str, str]] = []
        self.interview_complete = False

    def start_interview(self) -> Dict[str, Any]:
        """Begin the interview with a MedGemma opening question."""

        initial_prompt = (
            "আপনি একজন সহায়ক চিকিৎসা সহকারী যিনি প্রাক-পরিদর্শন তথ্য সংগ্রহ করছেন।\n"
            f"রোগীর নাম: {self.patient_data['name']}\n"
            f"রোগীর বয়স: {self.patient_data['age']}\n"
            f"রোগীর লিঙ্গ: {self.patient_data['gender']}\n\n"
            "উষ্ণ অভিবাদন দিয়ে শুরু করুন এবং তাদের আজকের ভিজিটের কারণ সম্পর্কে জিজ্ঞাসা করুন।"
        )

        response = generate_medgemma_response(
            prompt=initial_prompt,
            conversation_history=[],
            cache=self.cache,
        )

        self.conversation_history.append({"role": "assistant", "content": response})
        audio_base64 = generate_speech(response, cache=self.cache)

        return {"message": response, "audio": audio_base64, "complete": False}

    def process_user_response(self, user_message: str) -> Dict[str, Any]:
        """Process an incoming message and advance the conversation."""

        patient_context = create_patient_context(self.patient_data)
        patient_response = generate_gemini_response(
            prompt=patient_context,
            message=user_message,
            conversation_history=self.conversation_history,
            cache=self.cache,
        )

        self.conversation_history.append({"role": "user", "content": patient_response})

        follow_up_prompt = (
            "পূর্ববর্তী কথোপকথনের উপর ভিত্তি করে, বাংলায় একটি প্রাসঙ্গিক ফলো-আপ প্রশ্ন করুন।\n\n"
            f"রোগীর সর্বশেষ উত্তর: {patient_response}\n\n"
            "নিশ্চিত করুন যে আপনি উপসর্গের সূচনা, সময়কাল, তীব্রতা, সম্পর্কিত উপসর্গ, "
            "চিকিৎসা ইতিহাস এবং বর্তমান ওষুধ সম্পর্কে জানতে চেষ্টা করছেন। যদি পর্যাপ্ত তথ্য "
            "সংগ্রহ করা হয়ে থাকে (৫-৭টি বিনিময়ের পরে), বাংলায় বলুন \"তথ্য সংগ্রহ সম্পন্ন।\""
        )

        assistant_reply = generate_medgemma_response(
            prompt=follow_up_prompt,
            conversation_history=self.conversation_history,
            cache=self.cache,
        )

        self.conversation_history.append({"role": "assistant", "content": assistant_reply})

        if (
            self.COMPLETION_MARKER in assistant_reply
            or len(self.conversation_history) >= self.MAX_TURNS
        ):
            self.interview_complete = True

        audio_base64 = generate_speech(assistant_reply, cache=self.cache)

        return {
            "message": assistant_reply,
            "audio": audio_base64,
            "complete": self.interview_complete,
            "conversation": self.conversation_history,
        }

    def generate_report(self) -> str:
        """Create a Bangla medical summary using MedGemma."""

        return generate_medical_summary(
            patient_info=self.patient_data,
            conversation_history=self.conversation_history,
            cache=self.cache,
        )

    def get_transcript(self) -> str:
        """Return a formatted Bangla transcript."""

        lines = ["=== সাক্ষাৎকার ট্রান্সক্রিপ্ট ===", ""]
        for message in self.conversation_history:
            speaker = "চিকিৎসা সহকারী" if message["role"] == "assistant" else "রোগী"
            lines.append(f"{speaker}: {message['content']}")
            lines.append("")

        return "\n".join(lines)

