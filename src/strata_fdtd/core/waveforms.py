"""Audio waveform sources for FDTD simulation.

This module provides waveform classes that load audio from files
and provide them as source signals for FDTD acoustic simulation.

Classes:
    AudioFileWaveform: Load audio files (WAV, MP3, FLAC) as source waveforms

Example:
    >>> from strata_fdtd import AudioFileWaveform, CircularMembraneSource
    >>> waveform = AudioFileWaveform("test_tone.wav", amplitude=0.5)
    >>> source = CircularMembraneSource(
    ...     center=(0.1, 0.01, 0.15),
    ...     radius=0.1,
    ...     normal_axis='y',
    ...     waveform=waveform,
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.io import wavfile


@dataclass
class AudioFileWaveform:
    """Audio file source waveform for FDTD simulation.

    Loads audio from file, resamples to simulation sample rate,
    and provides pressure values at each timestep. Compatible with
    GaussianPulse interface for use with membrane sources.

    Args:
        filepath: Path to audio file (WAV natively, MP3/FLAC via soundfile)
        amplitude: Peak amplitude scaling (default 1.0)
        channel: Which channel to use for stereo files (0=left, 1=right, 'mix'=average)
        start_time: Start position in audio file (seconds)
        duration: Duration to use (None = entire file)
        loop: Whether to loop audio when end is reached

    Example:
        >>> waveform = AudioFileWaveform("test_tone.wav", amplitude=0.5)
        >>> # Use with membrane source
        >>> from strata_fdtd import CircularMembraneSource, GaussianPulse
        >>> source = CircularMembraneSource(
        ...     center=(0.1, 0.01, 0.15),
        ...     radius=0.1,
        ...     normal_axis='y',
        ...     waveform=waveform,
        ... )

    Note:
        FDTD simulations typically use very small timesteps (dt ~ 8 µs at 5mm
        resolution), corresponding to sample rates of ~120 kHz or higher.
        Audio files (typically 44.1/48 kHz) are automatically upsampled.
    """

    filepath: str | Path
    amplitude: float = 1.0
    channel: int | Literal["mix"] = "mix"
    start_time: float = 0.0
    duration: float | None = None
    loop: bool = False

    # Internal state (not part of public interface)
    _samples: NDArray[np.floating] | None = field(
        default=None, init=False, repr=False
    )
    _native_sr: int | None = field(default=None, init=False, repr=False)
    _resampled: dict[int, NDArray[np.floating]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Load and preprocess audio file."""
        self._load_audio()

    def _load_audio(self) -> None:
        """Load and preprocess audio file.

        Handles WAV files natively via scipy.io.wavfile.
        Other formats (MP3, FLAC, OGG) require the optional soundfile package.

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ImportError: If soundfile is needed but not installed
        """
        filepath = Path(self.filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        # Load audio using appropriate library
        if filepath.suffix.lower() == ".wav":
            sr, data = wavfile.read(filepath)
        else:
            # Use soundfile for other formats (MP3, FLAC, OGG, etc.)
            try:
                import soundfile as sf
            except ImportError as e:
                raise ImportError(
                    f"soundfile package required for {filepath.suffix} files. "
                    "Install with: pip install soundfile"
                ) from e
            data, sr = sf.read(filepath)

        self._native_sr = sr

        # Convert to float32 normalized to [-1, 1]
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128) / 128.0
        elif data.dtype in (np.float32, np.float64):
            data = data.astype(np.float32)
        else:
            # Attempt generic conversion
            data = data.astype(np.float32)

        # Handle stereo -> mono
        if len(data.shape) > 1:
            if self.channel == "mix":
                data = np.mean(data, axis=1)
            else:
                if self.channel >= data.shape[1]:
                    raise ValueError(
                        f"Channel {self.channel} requested but file only has "
                        f"{data.shape[1]} channels"
                    )
                data = data[:, self.channel]

        # Apply start/duration trimming
        start_sample = int(self.start_time * sr)
        if start_sample >= len(data):
            raise ValueError(
                f"start_time {self.start_time}s is beyond end of file "
                f"({len(data) / sr:.2f}s)"
            )

        if self.duration is not None:
            end_sample = start_sample + int(self.duration * sr)
            data = data[start_sample:end_sample]
        else:
            data = data[start_sample:]

        self._samples = data.astype(np.float32)

    def _resample_for_dt(self, dt: float) -> NDArray[np.floating]:
        """Resample audio to simulation timestep.

        Uses scipy.signal.resample for high-quality resampling.
        Results are cached by target sample rate.

        Args:
            dt: Simulation timestep in seconds

        Returns:
            Resampled audio array
        """
        target_sr = int(round(1.0 / dt))

        if target_sr in self._resampled:
            return self._resampled[target_sr]

        if target_sr == self._native_sr:
            resampled = self._samples
        else:
            # Resample using scipy's polyphase resampler
            num_samples = int(len(self._samples) * target_sr / self._native_sr)
            resampled = signal.resample(self._samples, num_samples).astype(np.float32)

        self._resampled[target_sr] = resampled
        return resampled

    def waveform(
        self, t: NDArray[np.floating], dt: float
    ) -> NDArray[np.floating]:
        """Get waveform values at specified times.

        Interface matches GaussianPulse.waveform() for compatibility
        with membrane sources and other FDTD components.

        Args:
            t: Array of time values in seconds
            dt: Simulation timestep in seconds

        Returns:
            Pressure values at each time, scaled by amplitude
        """
        resampled = self._resample_for_dt(dt)

        # Convert times to sample indices
        indices = (t / dt).astype(np.int64)

        if self.loop:
            # Wrap indices for looping
            indices = indices % len(resampled)
            return self.amplitude * resampled[indices]
        else:
            # Clamp to valid range, return 0 past end
            valid = (indices >= 0) & (indices < len(resampled))
            result = np.zeros_like(t, dtype=np.float32)
            result[valid] = resampled[indices[valid]]
            return self.amplitude * result

    @property
    def duration_seconds(self) -> float:
        """Total duration of loaded audio in seconds."""
        if self._samples is None or self._native_sr is None:
            return 0.0
        return len(self._samples) / self._native_sr

    @property
    def native_sample_rate(self) -> int:
        """Original sample rate of audio file."""
        return self._native_sr or 0

    @property
    def num_samples(self) -> int:
        """Number of samples in loaded audio."""
        if self._samples is None:
            return 0
        return len(self._samples)
