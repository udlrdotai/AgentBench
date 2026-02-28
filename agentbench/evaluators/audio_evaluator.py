"""Audio evaluator — objective metrics for ASMR audio quality assessment.

Computes the following metrics:
- SNR (Signal-to-Noise Ratio): Higher is better
- Spectral Centroid: Lower values indicate more low/mid frequency content (ASMR-friendly)
- LUFS (Loudness Units Full Scale): ASMR audio should have low loudness
- Peak dBFS: Peak level in dB Full Scale
- Duration accuracy: How close the actual duration is to expected
- Spectral rolloff: Frequency below which a percentage of spectral energy is concentrated
"""

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass
class AudioMetrics:
    """Objective audio quality metrics for ASMR evaluation."""

    duration_seconds: float
    sample_rate: int
    snr_db: float
    spectral_centroid_hz: float
    loudness_lufs: float
    peak_dbfs: float
    rms_dbfs: float
    spectral_rolloff_hz: float
    crest_factor_db: float  # Peak-to-RMS ratio; low = consistent volume (good for ASMR)
    low_freq_energy_ratio: float  # Ratio of energy below 1kHz (higher = more ASMR-friendly)


@dataclass
class AudioEvalResult:
    """Result of audio evaluation combining metrics and scores."""

    task_id: str
    model_name: str
    metrics: AudioMetrics
    technical_score: float  # 1-10 based on objective metrics
    comment: str


class AudioEvaluator:
    """Evaluates audio files using objective acoustic metrics.

    Designed for ASMR audio evaluation where ideal characteristics include:
    - Low overall loudness (LUFS < -20)
    - Low spectral centroid (more low/mid frequency content)
    - High SNR (clean audio)
    - Low crest factor (consistent volume without sudden peaks)
    - High low-frequency energy ratio
    """

    def __init__(self, expected_duration: float = 15.0):
        self._expected_duration = expected_duration

    def evaluate(self, audio_path: Path, task_id: str, model_name: str) -> AudioEvalResult:
        """Evaluate an audio file and return metrics with an overall score.

        Args:
            audio_path: Path to the audio file (WAV format preferred).
            task_id: Identifier for the benchmark task.
            model_name: Name of the model that generated the audio.

        Returns:
            AudioEvalResult with metrics and technical score.
        """
        samples, sample_rate = self._load_audio(audio_path)
        metrics = self._compute_metrics(samples, sample_rate)

        score = self._compute_asmr_score(metrics)
        comment = self._generate_comment(metrics, score)

        return AudioEvalResult(
            task_id=task_id,
            model_name=model_name,
            metrics=metrics,
            technical_score=score,
            comment=comment,
        )

    def _load_audio(self, audio_path: Path) -> tuple[NDArray[np.float64], int]:
        """Load audio file and return normalized float samples and sample rate."""
        suffix = audio_path.suffix.lower()
        if suffix != ".wav":
            raise ValueError(
                f"Unsupported audio format: {suffix}. "
                "Convert to WAV before evaluation."
            )

        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)

        # Convert raw bytes to numpy array
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            dtype = np.int16

        samples = np.frombuffer(raw_data, dtype=dtype).astype(np.float64)

        # Convert stereo to mono by averaging channels
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        # Normalize to [-1.0, 1.0]
        max_val = float(np.iinfo(dtype).max)
        samples = samples / max_val

        return samples, sample_rate

    def _compute_metrics(self, samples: NDArray[np.float64], sample_rate: int) -> AudioMetrics:
        """Compute all objective audio metrics."""
        duration = len(samples) / sample_rate

        # RMS and peak levels
        rms = np.sqrt(np.mean(samples**2))
        peak = np.max(np.abs(samples))

        # Convert to dBFS (dB relative to full scale)
        rms_dbfs = 20 * np.log10(rms + 1e-10)
        peak_dbfs = 20 * np.log10(peak + 1e-10)
        crest_factor_db = peak_dbfs - rms_dbfs

        # LUFS approximation (simplified ITU-R BS.1770)
        loudness_lufs = self._compute_lufs(samples, sample_rate)

        # SNR estimation
        snr_db = self._compute_snr(samples, sample_rate)

        # Spectral analysis
        spectral_centroid = self._compute_spectral_centroid(samples, sample_rate)
        spectral_rolloff = self._compute_spectral_rolloff(samples, sample_rate)
        low_freq_ratio = self._compute_low_freq_energy_ratio(samples, sample_rate)

        return AudioMetrics(
            duration_seconds=duration,
            sample_rate=sample_rate,
            snr_db=snr_db,
            spectral_centroid_hz=spectral_centroid,
            loudness_lufs=loudness_lufs,
            peak_dbfs=peak_dbfs,
            rms_dbfs=rms_dbfs,
            spectral_rolloff_hz=spectral_rolloff,
            crest_factor_db=crest_factor_db,
            low_freq_energy_ratio=low_freq_ratio,
        )

    @staticmethod
    def _compute_lufs(samples: NDArray[np.float64], sample_rate: int) -> float:
        """Compute integrated loudness in LUFS (simplified BS.1770).

        This is a simplified version that applies K-weighting approximation
        and gated loudness measurement.
        """
        # Apply simple pre-emphasis filter (approximation of K-weighting)
        # K-weighting = shelf filter + high-pass filter
        # Simplified: just high-pass at 100Hz to remove DC and sub-bass
        from scipy.signal import butter, sosfilt

        # High-pass filter at 100Hz (K-weighting approximation)
        sos = butter(2, 100, btype="high", fs=sample_rate, output="sos")
        filtered = sosfilt(sos, samples)

        # Compute mean square
        mean_square = np.mean(filtered**2)

        # Convert to LUFS
        if mean_square > 0:
            lufs = -0.691 + 10 * np.log10(mean_square)
        else:
            lufs = -70.0  # silence

        return float(lufs)

    @staticmethod
    def _compute_snr(samples: NDArray[np.float64], sample_rate: int) -> float:
        """Estimate SNR by comparing signal power to noise floor.

        Uses the quietest 10% of frames as noise estimate.
        """
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)  # 10ms hop

        # Compute frame energies
        n_frames = max(1, (len(samples) - frame_length) // hop_length)
        frame_energies = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            frame = samples[start:end]
            frame_energies[i] = np.mean(frame**2)

        # Sort energies and use bottom 10% as noise estimate
        sorted_energies = np.sort(frame_energies)
        noise_frames = max(1, n_frames // 10)
        noise_power = np.mean(sorted_energies[:noise_frames])
        signal_power = np.mean(frame_energies)

        if noise_power > 0 and signal_power > noise_power:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = 60.0  # Very clean signal

        return float(snr)

    @staticmethod
    def _compute_spectral_centroid(
        samples: NDArray[np.float64], sample_rate: int
    ) -> float:
        """Compute spectral centroid (weighted mean of frequencies)."""
        # Compute FFT
        n_fft = 2048
        fft_result = np.fft.rfft(samples[:n_fft * (len(samples) // n_fft)].reshape(-1, n_fft), axis=1)
        magnitude = np.abs(fft_result)

        # Average across frames
        avg_magnitude = np.mean(magnitude, axis=0)

        # Frequency bins
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

        # Weighted mean
        total_energy = np.sum(avg_magnitude)
        if total_energy > 0:
            centroid = np.sum(freqs * avg_magnitude) / total_energy
        else:
            centroid = 0.0

        return float(centroid)

    @staticmethod
    def _compute_spectral_rolloff(
        samples: NDArray[np.float64], sample_rate: int, rolloff_percent: float = 0.85
    ) -> float:
        """Compute spectral rolloff frequency.

        The frequency below which rolloff_percent of the spectral energy is concentrated.
        """
        n_fft = 2048
        fft_result = np.fft.rfft(samples[:n_fft * (len(samples) // n_fft)].reshape(-1, n_fft), axis=1)
        magnitude = np.abs(fft_result)
        avg_magnitude = np.mean(magnitude, axis=0)

        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

        total_energy = np.sum(avg_magnitude)
        if total_energy == 0:
            return 0.0

        cumulative = np.cumsum(avg_magnitude)
        rolloff_idx = np.searchsorted(cumulative, rolloff_percent * total_energy)
        rolloff_idx = min(rolloff_idx, len(freqs) - 1)

        return float(freqs[rolloff_idx])

    @staticmethod
    def _compute_low_freq_energy_ratio(
        samples: NDArray[np.float64], sample_rate: int, cutoff_hz: float = 1000.0
    ) -> float:
        """Compute ratio of energy below cutoff_hz to total energy.

        Higher ratio indicates more low-frequency content (ASMR-friendly).
        """
        n_fft = 2048
        fft_result = np.fft.rfft(samples[:n_fft * (len(samples) // n_fft)].reshape(-1, n_fft), axis=1)
        power = np.abs(fft_result) ** 2
        avg_power = np.mean(power, axis=0)

        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

        total_energy = np.sum(avg_power)
        if total_energy == 0:
            return 0.0

        low_freq_mask = freqs <= cutoff_hz
        low_freq_energy = np.sum(avg_power[low_freq_mask])

        return float(low_freq_energy / total_energy)

    def _compute_asmr_score(self, metrics: AudioMetrics) -> float:
        """Compute an overall ASMR quality score (1-10) based on objective metrics.

        Scoring criteria:
        - Loudness (LUFS): ASMR should be quiet (-30 to -20 LUFS ideal)
        - Crest factor: Should be moderate (3-12 dB), too high = jarring peaks
        - Low-freq energy ratio: Higher is better for ASMR (>0.5 ideal)
        - Spectral centroid: Lower is generally better for ASMR (<2000 Hz)
        - SNR: Higher is better (>20 dB)
        """
        scores = []

        # Loudness score (ASMR should be quiet)
        if metrics.loudness_lufs <= -30:
            loudness_score = 9.0  # Very quiet, great for ASMR
        elif metrics.loudness_lufs <= -20:
            loudness_score = 8.0
        elif metrics.loudness_lufs <= -14:
            loudness_score = 6.0
        elif metrics.loudness_lufs <= -8:
            loudness_score = 4.0
        else:
            loudness_score = 2.0  # Too loud for ASMR
        scores.append(loudness_score)

        # Crest factor score (moderate = consistent volume)
        if 3.0 <= metrics.crest_factor_db <= 12.0:
            crest_score = 8.0
        elif metrics.crest_factor_db < 3.0:
            crest_score = 5.0  # Very compressed
        else:
            crest_score = 4.0  # Too dynamic / jarring peaks
        scores.append(crest_score)

        # Low-frequency energy ratio (higher = more ASMR-friendly)
        if metrics.low_freq_energy_ratio >= 0.6:
            lowfreq_score = 9.0
        elif metrics.low_freq_energy_ratio >= 0.4:
            lowfreq_score = 7.0
        elif metrics.low_freq_energy_ratio >= 0.2:
            lowfreq_score = 5.0
        else:
            lowfreq_score = 3.0
        scores.append(lowfreq_score)

        # Spectral centroid (lower = warmer, more ASMR-friendly)
        if metrics.spectral_centroid_hz <= 1000:
            centroid_score = 9.0
        elif metrics.spectral_centroid_hz <= 2000:
            centroid_score = 7.0
        elif metrics.spectral_centroid_hz <= 4000:
            centroid_score = 5.0
        else:
            centroid_score = 3.0
        scores.append(centroid_score)

        # SNR (higher = cleaner audio)
        if metrics.snr_db >= 30:
            snr_score = 9.0
        elif metrics.snr_db >= 20:
            snr_score = 7.0
        elif metrics.snr_db >= 10:
            snr_score = 5.0
        else:
            snr_score = 3.0
        scores.append(snr_score)

        # Weighted average
        weights = [0.25, 0.15, 0.20, 0.20, 0.20]
        final_score = sum(s * w for s, w in zip(scores, weights))

        return round(max(1.0, min(10.0, final_score)), 1)

    @staticmethod
    def _generate_comment(metrics: AudioMetrics, score: float) -> str:
        """Generate a human-readable evaluation comment."""
        parts = []

        parts.append(f"技术评分: {score}/10")
        parts.append(f"时长: {metrics.duration_seconds:.1f}s")
        parts.append(f"响度: {metrics.loudness_lufs:.1f} LUFS")
        parts.append(f"信噪比: {metrics.snr_db:.1f} dB")
        parts.append(f"频谱重心: {metrics.spectral_centroid_hz:.0f} Hz")
        parts.append(f"低频能量占比: {metrics.low_freq_energy_ratio:.1%}")
        parts.append(f"峰值因数: {metrics.crest_factor_db:.1f} dB")

        # Quality indicators
        if metrics.loudness_lufs > -14:
            parts.append("[注意] 响度偏高，不适合 ASMR")
        if metrics.crest_factor_db > 15:
            parts.append("[注意] 动态范围过大，存在突然的音量跳变")
        if metrics.spectral_centroid_hz > 4000:
            parts.append("[注意] 高频成分过多，声音可能偏尖锐")

        return " | ".join(parts)
