"""MiniMax Music 2.5 model adapter for ASMR music/ambient generation."""

import time
import uuid
import wave
from pathlib import Path

import requests

from .base import AudioModel, AudioResult

MINIMAX_API_BASE = "https://api.minimax.chat/v1"

# Style prefix to steer music-2.5 toward ASMR ambient texture instead of songs
_ASMR_STYLE_PREFIX = (
    "ASMR ambient soundscape, no vocals, no singing, no melody, no beats. "
    "Ultra-quiet, minimal, sleep-inducing background texture. "
)


class MiniMaxMusicModel(AudioModel):
    """MiniMax Music 2.5 model for ASMR ambient music generation.

    Uses MiniMax's music-2.5 API to create calming ambient music
    and environmental soundscapes from lyrics + style prompt.
    Best suited for: ambient music ASMR, environmental atmosphere ASMR.
    """

    supported_asmr_types = {"ambient", "music"}

    def __init__(self, api_key: str | None = None, group_id: str | None = None):
        super().__init__("music-2.5")
        self._api_key = api_key
        self._group_id = group_id or ""

    def generate_audio(self, prompt: str, output_dir: Path, task_id: str = "", asmr_type: str = "") -> AudioResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"minimax_music_{task_id or uuid.uuid4().hex[:8]}.mp3"
        file_path = output_dir / file_name

        start_time = time.time()

        # Build style description: prepend ASMR guidance to the task prompt
        style_prompt = _ASMR_STYLE_PREFIX + prompt

        # Call MiniMax Music Generation API (music-2.5)
        url = f"{MINIMAX_API_BASE}/music_generation"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "music-2.5",
            "lyrics": "[Inst]\n[Break]\n[Inst]\n[Break]\n[Inst]",
            "prompt": style_prompt,
            "output_format": "url",
            "audio_setting": {
                "sample_rate": 44100,
                "bitrate": 128000,
                "format": "mp3",
            },
        }
        if self._group_id:
            params = {"GroupId": self._group_id}
        else:
            params = {}

        response = requests.post(url, headers=headers, json=payload, params=params, timeout=300)
        response.raise_for_status()

        result = response.json()

        # Check for API-level errors first
        base_resp = result.get("base_resp") or {}
        if base_resp.get("status_code", 0) != 0:
            raise RuntimeError(
                f"MiniMax API error: [{base_resp.get('status_code')}] "
                f"{base_resp.get('status_msg', 'unknown error')}"
            )

        # Extract audio URL from response and download
        data = result.get("data") or {}
        audio_url = data.get("audio", "")
        if not audio_url:
            audio_url = data.get("audio_file", "")

        if not audio_url:
            raise RuntimeError(f"MiniMax response did not contain audio URL: {result}")

        # Download the audio file
        audio_response = requests.get(audio_url, timeout=60)
        audio_response.raise_for_status()

        file_path.write_bytes(audio_response.content)

        generation_time = time.time() - start_time

        # Get audio metadata
        duration, sample_rate = self._get_audio_info(file_path)

        return AudioResult(
            file_path=file_path,
            duration_seconds=duration,
            sample_rate=sample_rate,
            format="mp3",
            model_name=self.name,
            task_id=task_id,
            generation_time_seconds=generation_time,
        )

    @staticmethod
    def _get_audio_info(file_path: Path) -> tuple[float, int]:
        """Get duration and sample rate from an audio file.

        For MP3 files, uses a rough estimation. For WAV files, reads headers.
        """
        suffix = file_path.suffix.lower()
        if suffix == ".wav":
            with wave.open(str(file_path), "rb") as wf:
                sample_rate = wf.getframerate()
                duration = wf.getnframes() / sample_rate
                return duration, sample_rate

        # For MP3/other formats, estimate from file size
        # A more accurate approach would use a library like mutagen
        file_size = file_path.stat().st_size
        # Rough estimate: 128kbps MP3 = 16KB/s
        estimated_duration = file_size / 16000.0
        return estimated_duration, 44100  # assume standard sample rate

    @property
    def name(self) -> str:
        return "minimax/music-2.5"
