"""Crossover filters for multi-driver speaker simulation.

This module provides crossover filter classes that split audio signals
into frequency bands for woofers, midrange drivers, and tweeters.

Classes:
    FilteredWaveform: Waveform with applied filter (lazy evaluation)
    Crossover: Two-way crossover with lowpass/highpass outputs
    ThreeWayCrossover: Three-way crossover for woofer/midrange/tweeter

Example:
    >>> from strata_fdtd import AudioFileWaveform, Crossover
    >>> audio = AudioFileWaveform("song.wav")
    >>> xover = Crossover(frequency=2000, order=4, type="linkwitz-riley")
    >>> woofer_signal = xover.lowpass(audio)
    >>> tweeter_signal = xover.highpass(audio)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sig


@runtime_checkable
class WaveformProtocol(Protocol):
    """Protocol for waveform sources compatible with FDTD simulation."""

    def waveform(
        self, t: NDArray[np.floating], dt: float
    ) -> NDArray[np.floating]: ...


FilterType = Literal["lowpass", "highpass", "bandpass"]
FilterDesign = Literal["butterworth", "linkwitz-riley", "bessel"]


@dataclass
class FilteredWaveform:
    """Waveform with applied digital filter.

    Lazy evaluation - filter is applied when waveform() is called.
    Results are cached per sample rate for efficiency.

    Args:
        source: Source waveform (AudioFileWaveform or any WaveformProtocol)
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
        frequency: Cutoff frequency in Hz (single value for LP/HP, tuple for BP)
        order: Filter order (2, 4, 6, 8 typical)
        design: Filter design ('butterworth', 'linkwitz-riley', 'bessel')

    Example:
        >>> from strata_fdtd import AudioFileWaveform
        >>> audio = AudioFileWaveform("test.wav")
        >>> filtered = FilteredWaveform(
        ...     source=audio,
        ...     filter_type="lowpass",
        ...     frequency=1000,
        ...     order=4,
        ...     design="linkwitz-riley",
        ... )
        >>> # Use with membrane source
        >>> source = CircularMembraneSource(..., waveform=filtered)
    """

    source: WaveformProtocol | Any
    filter_type: FilterType
    frequency: float | tuple[float, float]
    order: int = 4
    design: FilterDesign = "linkwitz-riley"

    # Cache for filtered results (keyed by sample rate)
    _filtered_cache: dict[int, NDArray[np.floating]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _design_filter(
        self, sample_rate: int
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Design filter coefficients for given sample rate.

        Args:
            sample_rate: Target sample rate in Hz

        Returns:
            Tuple of (b, a) filter coefficients

        Raises:
            ValueError: If filter frequency exceeds Nyquist limit
        """
        nyq = sample_rate / 2

        # Normalize frequency to Nyquist
        if isinstance(self.frequency, tuple):
            # Bandpass
            normalized_freq = (self.frequency[0] / nyq, self.frequency[1] / nyq)
            if normalized_freq[0] >= 1.0 or normalized_freq[1] >= 1.0:
                raise ValueError(
                    f"Filter frequencies {self.frequency} exceed Nyquist "
                    f"frequency {nyq} Hz for sample rate {sample_rate} Hz"
                )
        else:
            normalized_freq = self.frequency / nyq
            if normalized_freq >= 1.0:
                raise ValueError(
                    f"Filter frequency {self.frequency} Hz exceeds Nyquist "
                    f"frequency {nyq} Hz for sample rate {sample_rate} Hz"
                )

        # Design filter based on type
        if self.design == "linkwitz-riley":
            # Linkwitz-Riley is cascaded Butterworth at half order
            # LR4 = two cascaded 2nd-order Butterworth
            half_order = max(1, self.order // 2)
            b, a = sig.butter(half_order, normalized_freq, btype=self.filter_type)
            return b, a
        elif self.design == "butterworth":
            return sig.butter(self.order, normalized_freq, btype=self.filter_type)
        elif self.design == "bessel":
            return sig.bessel(
                self.order, normalized_freq, btype=self.filter_type, norm="phase"
            )
        else:
            raise ValueError(f"Unknown filter design: {self.design}")

    def _apply_filter(
        self, samples: NDArray[np.floating], sample_rate: int
    ) -> NDArray[np.floating]:
        """Apply filter to sample array.

        Uses forward-backward filtering (filtfilt) for zero phase distortion.
        Linkwitz-Riley applies the filter twice for the cascaded response.

        Args:
            samples: Input sample array
            sample_rate: Sample rate in Hz

        Returns:
            Filtered sample array
        """
        b, a = self._design_filter(sample_rate)

        # Ensure we have enough samples for filtfilt
        padlen = 3 * max(len(b), len(a))
        if len(samples) <= padlen:
            # For very short signals, use lfilter with appropriate padding
            padded = np.pad(samples, (padlen, padlen), mode="edge")
            if self.design == "linkwitz-riley":
                # Apply twice for LR (cascaded Butterworth)
                filtered = sig.lfilter(b, a, padded)
                filtered = sig.lfilter(b, a, filtered)
            else:
                filtered = sig.lfilter(b, a, padded)
            return filtered[padlen:-padlen].astype(np.float32)

        if self.design == "linkwitz-riley":
            # Apply twice for LR (cascaded Butterworth = LR response)
            filtered = sig.filtfilt(b, a, samples)
            filtered = sig.filtfilt(b, a, filtered)
        else:
            # Forward-backward for zero phase
            filtered = sig.filtfilt(b, a, samples)

        return filtered.astype(np.float32)

    def _get_source_samples(self, dt: float) -> NDArray[np.floating]:
        """Get samples from source waveform.

        Args:
            dt: Simulation timestep in seconds

        Returns:
            Source samples resampled to simulation rate
        """
        # Check if source has _resample_for_dt (AudioFileWaveform)
        if hasattr(self.source, "_resample_for_dt"):
            return self.source._resample_for_dt(dt)

        # Fallback: generate samples using waveform method
        # This works for other waveform types but may be slower
        sample_rate = int(round(1.0 / dt))
        duration = getattr(self.source, "duration_seconds", 1.0)
        num_samples = int(duration * sample_rate)
        t = np.arange(num_samples) * dt
        return self.source.waveform(t, dt)

    def waveform(
        self, t: NDArray[np.floating], dt: float
    ) -> NDArray[np.floating]:
        """Get filtered waveform values at specified times.

        Interface matches GaussianPulse.waveform() for compatibility
        with membrane sources and other FDTD components.

        Args:
            t: Array of time values in seconds
            dt: Simulation timestep in seconds

        Returns:
            Filtered pressure values at each time
        """
        sample_rate = int(round(1.0 / dt))

        # Check cache
        if sample_rate not in self._filtered_cache:
            # Get source samples and filter
            source_samples = self._get_source_samples(dt)
            self._filtered_cache[sample_rate] = self._apply_filter(
                source_samples, sample_rate
            )

        filtered = self._filtered_cache[sample_rate]

        # Convert times to sample indices
        indices = (t / dt).astype(np.int64)

        # Handle bounds
        valid = (indices >= 0) & (indices < len(filtered))
        result = np.zeros_like(t, dtype=np.float32)
        result[valid] = filtered[indices[valid]]

        # Apply amplitude if source has it
        amplitude = getattr(self.source, "amplitude", 1.0)
        return amplitude * result

    @property
    def duration_seconds(self) -> float:
        """Duration of source waveform in seconds."""
        return getattr(self.source, "duration_seconds", 0.0)


@dataclass
class Crossover:
    """Two-way crossover filter for speaker simulation.

    Creates lowpass and highpass filtered versions of a waveform
    at a specified crossover frequency.

    Supports three filter designs:
    - Butterworth: Maximally flat passband
    - Linkwitz-Riley: Acoustically flat sum (recommended for audio)
    - Bessel: Linear phase, good transient response

    Args:
        frequency: Crossover frequency in Hz
        order: Filter order (2, 4, 6, 8 typical; default 4)
        type: Filter design type (default 'linkwitz-riley')

    Example:
        >>> from strata_fdtd import AudioFileWaveform, Crossover
        >>> audio = AudioFileWaveform("music.wav")
        >>> xover = Crossover(frequency=2500, order=4, type='linkwitz-riley')
        >>> woofer_signal = xover.lowpass(audio)
        >>> tweeter_signal = xover.highpass(audio)
        >>>
        >>> # Assign to drivers
        >>> enc.add_driver(..., name="woofer", waveform=woofer_signal)
        >>> enc.add_driver(..., name="tweeter", waveform=tweeter_signal)

    Note:
        Linkwitz-Riley crossovers sum to unity gain at all frequencies,
        making them ideal for multi-driver speaker systems. The outputs
        are in-phase at the crossover point.
    """

    frequency: float
    order: int = 4
    type: FilterDesign = "linkwitz-riley"

    def lowpass(self, waveform: WaveformProtocol | Any) -> FilteredWaveform:
        """Create lowpass-filtered version of waveform.

        Args:
            waveform: Source waveform to filter

        Returns:
            FilteredWaveform configured for lowpass filtering
        """
        return FilteredWaveform(
            source=waveform,
            filter_type="lowpass",
            frequency=self.frequency,
            order=self.order,
            design=self.type,
        )

    def highpass(self, waveform: WaveformProtocol | Any) -> FilteredWaveform:
        """Create highpass-filtered version of waveform.

        Args:
            waveform: Source waveform to filter

        Returns:
            FilteredWaveform configured for highpass filtering
        """
        return FilteredWaveform(
            source=waveform,
            filter_type="highpass",
            frequency=self.frequency,
            order=self.order,
            design=self.type,
        )


@dataclass
class ThreeWayCrossover:
    """Three-way crossover for woofer/midrange/tweeter systems.

    Creates three frequency bands from a single source waveform:
    - Woofer: frequencies below low_freq
    - Midrange: frequencies between low_freq and high_freq
    - Tweeter: frequencies above high_freq

    Args:
        low_freq: Low/mid crossover frequency in Hz
        high_freq: Mid/high crossover frequency in Hz
        order: Filter order (default 4)
        type: Filter design type (default 'linkwitz-riley')

    Example:
        >>> audio = AudioFileWaveform("orchestra.wav")
        >>> xover = ThreeWayCrossover(
        ...     low_freq=500,
        ...     high_freq=4000,
        ...     order=4,
        ...     type='linkwitz-riley'
        ... )
        >>> enc.add_driver(..., name="woofer", waveform=xover.woofer(audio))
        >>> enc.add_driver(..., name="midrange", waveform=xover.midrange(audio))
        >>> enc.add_driver(..., name="tweeter", waveform=xover.tweeter(audio))

    Note:
        The midrange band is created using a bandpass filter rather than
        cascading high and low pass filters, which provides cleaner response.
    """

    low_freq: float
    high_freq: float
    order: int = 4
    type: FilterDesign = "linkwitz-riley"

    def __post_init__(self) -> None:
        """Validate crossover frequencies."""
        if self.low_freq >= self.high_freq:
            raise ValueError(
                f"low_freq ({self.low_freq}) must be less than "
                f"high_freq ({self.high_freq})"
            )

    def woofer(self, waveform: WaveformProtocol | Any) -> FilteredWaveform:
        """Create lowpass-filtered signal for woofer.

        Args:
            waveform: Source waveform to filter

        Returns:
            FilteredWaveform with frequencies below low_freq
        """
        return FilteredWaveform(
            source=waveform,
            filter_type="lowpass",
            frequency=self.low_freq,
            order=self.order,
            design=self.type,
        )

    def midrange(self, waveform: WaveformProtocol | Any) -> FilteredWaveform:
        """Create bandpass-filtered signal for midrange driver.

        Args:
            waveform: Source waveform to filter

        Returns:
            FilteredWaveform with frequencies between low_freq and high_freq
        """
        return FilteredWaveform(
            source=waveform,
            filter_type="bandpass",
            frequency=(self.low_freq, self.high_freq),
            order=self.order,
            design=self.type,
        )

    def tweeter(self, waveform: WaveformProtocol | Any) -> FilteredWaveform:
        """Create highpass-filtered signal for tweeter.

        Args:
            waveform: Source waveform to filter

        Returns:
            FilteredWaveform with frequencies above high_freq
        """
        return FilteredWaveform(
            source=waveform,
            filter_type="highpass",
            frequency=self.high_freq,
            order=self.order,
            design=self.type,
        )
