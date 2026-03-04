"""Google Gemini TTS model adapter for ASMR audio generation."""

import logging
import time
import uuid
import wave
from pathlib import Path

from google import genai
from google.genai import types

from .base import AudioModel, AudioResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2  # seconds, doubled each retry

# Prompt prefix for voice-based ASMR (whisper / roleplay).
# Instructs the model on delivery style without saying "Generate" —
# the task prompt already describes what to say.
_PROMPT_PREFIX_VOICE = (
    "Use a very soft, gentle, whispering ASMR tone. "
    "Speak slowly with natural pauses and soft breathing. "
)


class GeminiTTSModel(AudioModel):
    """Google Gemini TTS model for ASMR voice generation.

    Uses Gemini's native audio output capability with controlled voice style.
    Best suited for: whisper voice ASMR, roleplay ASMR.
    """

    supported_asmr_types = {"whisper", "roleplay"}

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "gemini-2.5-flash-preview-tts",
        voice: str = "Kore",
    ):
        super().__init__(model_id)
        self._client = genai.Client(api_key=api_key)
        self._voice = voice

    def _call_api(self, prompt: str, asmr_type: str = "") -> "genai.types.GenerateContentResponse":
        """Call Gemini API with retry logic for transient errors."""
        # Add voice-style prefix for speech-based ASMR types
        if asmr_type in ("whisper", "roleplay"):
            full_prompt = _PROMPT_PREFIX_VOICE + prompt
        else:
            full_prompt = prompt

        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.models.generate_content(
                    model=self.model_id,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=self._voice,
                                ),
                            ),
                        ),
                    ),
                )
                return response
            except Exception as e:
                last_error = e
                err_str = str(e)
                # Retry on 500/503 server errors
                if "500" in err_str or "503" in err_str or "INTERNAL" in err_str:
                    wait = _RETRY_BACKOFF * (2 ** attempt)
                    logger.warning("Gemini API error (attempt %d/%d): %s — retrying in %ds", attempt + 1, _MAX_RETRIES, err_str, wait)
                    time.sleep(wait)
                else:
                    raise
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _extract_audio(response) -> tuple[bytes, int]:
        """Extract audio bytes and sample rate from Gemini response.

        Raises RuntimeError if the response doesn't contain audio data.
        """
        candidates = getattr(response, "candidates", None)
        if not candidates:
            raise RuntimeError("Gemini response contained no candidates")

        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            # Include finish_reason / safety info for debugging
            finish_reason = getattr(candidates[0], "finish_reason", "unknown")
            raise RuntimeError(f"Gemini response has no content parts (finish_reason={finish_reason})")

        sample_rate = 24000  # Gemini default
        for part in parts:
            if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                mime = part.inline_data.mime_type
                if "rate=" in mime:
                    sample_rate = int(mime.split("rate=")[1].split(";")[0])
                return part.inline_data.data, sample_rate

        raise RuntimeError("Gemini response did not contain audio data")

    def generate_audio(self, prompt: str, output_dir: Path, task_id: str = "", asmr_type: str = "") -> AudioResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"gemini_tts_{task_id or uuid.uuid4().hex[:8]}.wav"
        file_path = output_dir / file_name

        start_time = time.time()
        response = self._call_api(prompt, asmr_type=asmr_type)
        generation_time = time.time() - start_time

        audio_data, sample_rate = self._extract_audio(response)

        # Write raw PCM data as WAV file
        self._write_wav(file_path, audio_data, sample_rate)

        # Calculate duration
        # PCM 16-bit mono: each sample is 2 bytes
        num_samples = len(audio_data) // 2
        duration = num_samples / sample_rate

        return AudioResult(
            file_path=file_path,
            duration_seconds=duration,
            sample_rate=sample_rate,
            format="wav",
            model_name=self.name,
            task_id=task_id,
            generation_time_seconds=generation_time,
            metadata={"voice": self._voice},
        )

    @staticmethod
    def _write_wav(file_path: Path, pcm_data: bytes, sample_rate: int) -> None:
        """Write raw PCM data to a WAV file (16-bit mono)."""
        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)

    @property
    def name(self) -> str:
        return f"google/{self.model_id}"
