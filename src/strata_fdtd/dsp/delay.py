"""Time delay for driver alignment in speaker simulation.

This module provides waveform delay for time-aligning multiple drivers
in a speaker system.

Classes:
    DelayedWaveform: Waveform with time delay applied

Example:
    >>> from strata_fdtd import AudioFileWaveform, Crossover, DelayedWaveform
    >>> audio = AudioFileWaveform("song.wav")
    >>> xover = Crossover(frequency=2000)
    >>> # Delay tweeter by 2mm path length difference
    >>> tweeter_waveform = DelayedWaveform(
    ...     source=xover.highpass(audio),
    ...     delay_seconds=2e-3 / 343,  # 2mm at speed of sound
    ... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from strata_fdtd.dsp.crossover import WaveformProtocol


@dataclass
class DelayedWaveform:
    """Waveform with time delay for driver alignment.

    Real speakers often need time alignment because:
    - Tweeters are mounted in front of woofers
    - Voice coil depths differ between drivers
    - Acoustic centers don't align with baffle plane

    This class applies a simple time delay to any waveform source,
    allowing precise alignment of multiple drivers.

    Args:
        source: Source waveform (AudioFileWaveform, FilteredWaveform, etc.)
        delay_seconds: Time delay in seconds (positive = later)

    Example:
        >>> from strata_fdtd import AudioFileWaveform, Crossover, DelayedWaveform
        >>> audio = AudioFileWaveform("test.wav")
        >>> xover = Crossover(frequency=2000, order=4, type='linkwitz-riley')
        >>>
        >>> # Tweeter is 15mm ahead of woofer acoustic center
        >>> # Delay tweeter so sound arrives at listening position simultaneously
        >>> speed_of_sound = 343  # m/s
        >>> tweeter_delay = 0.015 / speed_of_sound  # ~44 microseconds
        >>>
        >>> tweeter_waveform = DelayedWaveform(
        ...     source=xover.highpass(audio),
        ...     delay_seconds=tweeter_delay,
        ... )
        >>> woofer_waveform = xover.lowpass(audio)
        >>>
        >>> enc.add_driver(..., name="tweeter", waveform=tweeter_waveform)
        >>> enc.add_driver(..., name="woofer", waveform=woofer_waveform)

    Note:
        In FDTD simulation, time alignment may be less critical than in
        real speakers because the simulation naturally handles propagation
        delays. However, aligning driver signals as in the physical system
        ensures accurate reproduction of the speaker's acoustic behavior.
    """

    source: WaveformProtocol | Any
    delay_seconds: float

    def waveform(
        self, t: NDArray[np.floating], dt: float
    ) -> NDArray[np.floating]:
        """Get delayed waveform values at specified times.

        For times before the delay, returns zero (signal hasn't arrived yet).
        Otherwise returns the source waveform value at (t - delay).

        Args:
            t: Array of time values in seconds
            dt: Simulation timestep in seconds

        Returns:
            Delayed pressure values at each time
        """
        # Calculate delayed time
        delayed_t = t - self.delay_seconds

        # For negative times (before delay), we'll get zeros from source
        # since most waveforms return 0 for t < 0
        # We clip to 0 to be safe with any waveform implementation
        delayed_t = np.maximum(delayed_t, 0.0)

        return self.source.waveform(delayed_t, dt)

    @property
    def duration_seconds(self) -> float:
        """Duration of delayed waveform in seconds.

        Includes the delay in total duration.
        """
        source_duration = getattr(self.source, "duration_seconds", 0.0)
        return source_duration + self.delay_seconds

    @property
    def delay_samples(self) -> int:
        """Get delay in samples (requires knowing dt)."""
        # This is informational only - actual sample rate depends on simulation
        raise NotImplementedError(
            "delay_samples requires sample rate. Use delay_seconds / dt."
        )
