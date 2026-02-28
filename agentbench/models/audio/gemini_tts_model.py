"""Google Gemini TTS model adapter for ASMR audio generation."""

import struct
import time
import uuid
import wave
from pathlib import Path

from google import genai
from google.genai import types

from .base import AudioModel, AudioResult

ASMR_SYSTEM_INSTRUCTION = (
    "You are an ASMR audio creator. Generate speech in a very soft, gentle, "
    "whispering tone. Speak slowly with natural pauses and soft breathing. "
    "Your voice should be calming, intimate, and sleep-inducing."
)


class GeminiTTSModel(AudioModel):
    """Google Gemini TTS model for ASMR voice generation.

    Uses Gemini's native audio output capability with controlled voice style.
    Best suited for: whisper voice ASMR, roleplay ASMR.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "gemini-2.5-flash",
        voice: str = "Kore",
    ):
        super().__init__(model_id)
        self._client = genai.Client(api_key=api_key)
        self._voice = voice

    def generate_audio(self, prompt: str, output_dir: Path, task_id: str = "") -> AudioResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"gemini_tts_{task_id or uuid.uuid4().hex[:8]}.wav"
        file_path = output_dir / file_name

        start_time = time.time()

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=ASMR_SYSTEM_INSTRUCTION,
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_id=self._voice,
                    ),
                ),
            ),
        )

        generation_time = time.time() - start_time

        # Extract audio data from response
        audio_data = None
        sample_rate = 24000  # Gemini default
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                audio_data = part.inline_data.data
                # Parse sample rate from mime type if available (e.g. "audio/pcm;rate=24000")
                mime = part.inline_data.mime_type
                if "rate=" in mime:
                    sample_rate = int(mime.split("rate=")[1].split(";")[0])
                break

        if audio_data is None:
            raise RuntimeError("Gemini response did not contain audio data")

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
