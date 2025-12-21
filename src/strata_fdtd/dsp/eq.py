"""Parametric equalizer for waveform shaping.

This module provides EQ processing for speaker simulation, allowing
frequency response shaping before driving membrane sources.

Classes:
    EQBand: Single EQ band configuration
    EQWaveform: Waveform with EQ applied
    ParametricEQ: Multi-band parametric equalizer

Example:
    >>> from strata_fdtd import AudioFileWaveform, ParametricEQ
    >>> audio = AudioFileWaveform("recording.wav")
    >>> eq = ParametricEQ()
    >>> eq.add_band(100, -3, 1.0, 'peaking')   # Cut 100Hz by 3dB
    >>> eq.add_band(3000, 2, 0.7, 'peaking')   # Boost 3kHz by 2dB
    >>> eq.add_band(80, 0, 0, 'highpass')       # 80Hz highpass
    >>> processed = eq.apply(audio)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sig

from strata_fdtd.dsp.crossover import WaveformProtocol

EQBandType = Literal["peaking", "lowshelf", "highshelf", "lowpass", "highpass"]


@dataclass
class EQBand:
    """Configuration for a single EQ band.

    Args:
        frequency: Center or corner frequency in Hz
        gain_db: Gain in dB (positive = boost, negative = cut)
        q: Q factor (bandwidth control; higher = narrower)
        type: Band type ('peaking', 'lowshelf', 'highshelf', 'lowpass', 'highpass')
    """

    frequency: float
    gain_db: float
    q: float
    type: EQBandType


@dataclass
class EQWaveform:
    """Waveform with parametric EQ applied.

    Lazy evaluation - EQ is applied when waveform() is called.
    Results are cached per sample rate for efficiency.

    Args:
        source: Source waveform to process
        bands: List of EQ bands to apply

    Note:
        EQ bands are applied in series. The order generally doesn't
        matter for linear filters, but processing happens in the
        order bands are defined.
    """

    source: WaveformProtocol | Any
    bands: list[EQBand]

    # Cache for processed results
    _processed_cache: dict[int, NDArray[np.floating]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _design_biquad(
        self, band: EQBand, sample_rate: int
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Design biquad filter coefficients for an EQ band.

        Uses the Audio EQ Cookbook formulas for biquad filter design.
        Reference: https://www.w3.org/2011/audio/audio-eq-cookbook.html

        Args:
            band: EQ band configuration
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (b, a) biquad coefficients
        """
        # Normalized frequency
        w0 = 2 * np.pi * band.frequency / sample_rate
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)

        # Amplitude from dB
        A = 10 ** (band.gain_db / 40)  # sqrt(10^(dB/20))

        if band.type in ("lowpass", "highpass"):
            # For LP/HP, use simple 2nd-order design
            alpha = sin_w0 / (2 * max(band.q, 0.1))  # Avoid div by zero

            if band.type == "lowpass":
                b0 = (1 - cos_w0) / 2
                b1 = 1 - cos_w0
                b2 = (1 - cos_w0) / 2
                a0 = 1 + alpha
                a1 = -2 * cos_w0
                a2 = 1 - alpha
            else:  # highpass
                b0 = (1 + cos_w0) / 2
                b1 = -(1 + cos_w0)
                b2 = (1 + cos_w0) / 2
                a0 = 1 + alpha
                a1 = -2 * cos_w0
                a2 = 1 - alpha

        elif band.type == "peaking":
            alpha = sin_w0 / (2 * max(band.q, 0.1))

            b0 = 1 + alpha * A
            b1 = -2 * cos_w0
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * cos_w0
            a2 = 1 - alpha / A

        elif band.type == "lowshelf":
            alpha = (
                sin_w0 / 2 * np.sqrt((A + 1 / A) * (1 / max(band.q, 0.1) - 1) + 2)
            )

            b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
            b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
            a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

        elif band.type == "highshelf":
            alpha = (
                sin_w0 / 2 * np.sqrt((A + 1 / A) * (1 / max(band.q, 0.1) - 1) + 2)
            )

            b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
            b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
            a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

        else:
            raise ValueError(f"Unknown EQ band type: {band.type}")

        # Normalize coefficients
        b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
        a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)

        return b, a

    def _apply_eq(
        self, samples: NDArray[np.floating], sample_rate: int
    ) -> NDArray[np.floating]:
        """Apply all EQ bands to sample array.

        Args:
            samples: Input sample array
            sample_rate: Sample rate in Hz

        Returns:
            EQ-processed sample array
        """
        result = samples.astype(np.float64)

        for band in self.bands:
            b, a = self._design_biquad(band, sample_rate)

            # Use filtfilt for zero-phase response
            padlen = 3 * max(len(b), len(a))
            if len(result) > padlen:
                result = sig.filtfilt(b, a, result)
            else:
                # For very short signals, use lfilter with padding
                padded = np.pad(result, (padlen, padlen), mode="edge")
                result = sig.lfilter(b, a, padded)[padlen:-padlen]

        return result.astype(np.float32)

    def _get_source_samples(self, dt: float) -> NDArray[np.floating]:
        """Get samples from source waveform."""
        if hasattr(self.source, "_resample_for_dt"):
            return self.source._resample_for_dt(dt)

        # Fallback for other waveform types
        sample_rate = int(round(1.0 / dt))
        duration = getattr(self.source, "duration_seconds", 1.0)
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) * dt
        return self.source.waveform(t, dt)

    def waveform(
        self, t: NDArray[np.floating], dt: float
    ) -> NDArray[np.floating]:
        """Get EQ-processed waveform values at specified times.

        Args:
            t: Array of time values in seconds
            dt: Simulation timestep in seconds

        Returns:
            EQ-processed pressure values at each time
        """
        sample_rate = int(round(1.0 / dt))

        # Check cache
        if sample_rate not in self._processed_cache:
            source_samples = self._get_source_samples(dt)
            self._processed_cache[sample_rate] = self._apply_eq(
                source_samples, sample_rate
            )

        processed = self._processed_cache[sample_rate]

        # Convert times to sample indices
        indices = (t / dt).astype(np.int64)

        # Handle bounds
        valid = (indices >= 0) & (indices < len(processed))
        result = np.zeros_like(t, dtype=np.float32)
        result[valid] = processed[indices[valid]]

        # Apply amplitude if source has it
        amplitude = getattr(self.source, "amplitude", 1.0)
        return amplitude * result

    @property
    def duration_seconds(self) -> float:
        """Duration of source waveform in seconds."""
        return getattr(self.source, "duration_seconds", 0.0)


@dataclass
class ParametricEQ:
    """Multi-band parametric equalizer for waveform processing.

    Builds a collection of EQ bands that can be applied to any
    waveform source. Useful for:
    - Compensating for driver response curves
    - Room correction simulation
    - Tonal shaping experiments

    Example:
        >>> from strata_fdtd import AudioFileWaveform, ParametricEQ
        >>> audio = AudioFileWaveform("test.wav")
        >>>
        >>> # Create EQ with common adjustments
        >>> eq = ParametricEQ()
        >>> eq.add_band(100, -3, 1.0, 'peaking')    # Cut bass boom
        >>> eq.add_band(3000, 2, 0.7, 'peaking')    # Add presence
        >>> eq.add_band(80, 0, 0.707, 'highpass')   # Subsonic filter
        >>> eq.add_band(12000, -2, 0.707, 'highshelf')  # Reduce harshness
        >>>
        >>> # Apply to waveform
        >>> processed = eq.apply(audio)

    Note:
        The Q parameter controls bandwidth:
        - Q=0.707: Butterworth (maximally flat)
        - Q=1.0: Moderate bandwidth
        - Q=2.0+: Narrow, surgical cuts/boosts
    """

    bands: list[EQBand] = field(default_factory=list)

    def add_band(
        self,
        frequency: float,
        gain_db: float,
        q: float,
        type: EQBandType,
    ) -> ParametricEQ:
        """Add an EQ band.

        Args:
            frequency: Center or corner frequency in Hz
            gain_db: Gain in dB (positive = boost, negative = cut)
            q: Q factor (bandwidth control)
            type: Band type

        Returns:
            self for method chaining

        Example:
            >>> eq = ParametricEQ()
            >>> eq.add_band(100, -3, 1.0, 'peaking')
            >>> eq.add_band(5000, 2, 0.7, 'peaking')
        """
        self.bands.append(
            EQBand(frequency=frequency, gain_db=gain_db, q=q, type=type)
        )
        return self

    def clear(self) -> ParametricEQ:
        """Remove all EQ bands.

        Returns:
            self for method chaining
        """
        self.bands.clear()
        return self

    def apply(self, waveform: WaveformProtocol | Any) -> EQWaveform:
        """Apply EQ to a waveform.

        Args:
            waveform: Source waveform to process

        Returns:
            EQWaveform with all configured bands applied
        """
        return EQWaveform(source=waveform, bands=list(self.bands))


# Convenience function for common EQ curves
def create_baffle_step_compensation(
    baffle_width: float,
    speed_of_sound: float = 343.0,
    boost_db: float = 3.0,
) -> ParametricEQ:
    """Create EQ to compensate for baffle step diffraction.

    When a speaker driver is mounted on a finite baffle, bass response
    drops by ~6dB below the baffle step frequency (where wavelength
    equals baffle width). This creates an EQ to compensate.

    Args:
        baffle_width: Baffle width in meters
        speed_of_sound: Speed of sound in m/s (default 343)
        boost_db: Amount of bass boost (default 3dB, half of 6dB step)

    Returns:
        ParametricEQ configured for baffle step compensation

    Example:
        >>> # 20cm wide baffle
        >>> eq = create_baffle_step_compensation(0.2)
        >>> compensated = eq.apply(audio)
    """
    # Baffle step frequency: f = c / (2 * width)
    f_step = speed_of_sound / (2 * baffle_width)

    eq = ParametricEQ()
    # Low shelf boost below baffle step frequency
    eq.add_band(f_step, boost_db, 0.707, "lowshelf")

    return eq
