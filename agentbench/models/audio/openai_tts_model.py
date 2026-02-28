"""OpenAI TTS model adapter (gpt-4o-mini-tts) for ASMR audio generation."""

import time
import uuid
import wave
from pathlib import Path

import openai

from .base import AudioModel, AudioResult

ASMR_VOICE_INSTRUCTIONS = (
    "Speak in a very soft, gentle whisper. Use an ASMR-style delivery with "
    "careful, slow pacing. Include natural pauses between phrases. Keep your "
    "voice intimate, calm, and soothing, as if speaking directly into the "
    "listener's ear. Breathe softly and naturally between sentences."
)


class OpenAITTSModel(AudioModel):
    """OpenAI TTS model using gpt-4o-mini-tts for ASMR voice generation.

    Best suited for: whisper voice ASMR, roleplay ASMR.
    Limited capability for: trigger sounds, environmental ambience, music.
    """

    def __init__(self, api_key: str | None = None, voice: str = "alloy"):
        super().__init__("gpt-4o-mini-tts")
        self._client = openai.OpenAI(api_key=api_key)
        self._voice = voice

    def generate_audio(self, prompt: str, output_dir: Path, task_id: str = "") -> AudioResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"openai_tts_{task_id or uuid.uuid4().hex[:8]}.wav"
        file_path = output_dir / file_name

        start_time = time.time()

        response = self._client.audio.speech.create(
            model=self.model_id,
            voice=self._voice,
            input=prompt,
            instructions=ASMR_VOICE_INSTRUCTIONS,
            response_format="wav",
        )
        response.stream_to_file(str(file_path))

        generation_time = time.time() - start_time

        # Read audio metadata from WAV file
        with wave.open(str(file_path), "rb") as wf:
            sample_rate = wf.getframerate()
            duration = wf.getnframes() / sample_rate

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

    @property
    def name(self) -> str:
        return "openai/gpt-4o-mini-tts"
