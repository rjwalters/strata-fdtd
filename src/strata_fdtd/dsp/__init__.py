"""DSP processing for multi-driver speaker simulation.

This module provides digital signal processing components for preparing
audio waveforms for FDTD simulation of multi-driver speaker systems.

Classes:
    Crossover: Two-way crossover filter (lowpass/highpass)
    ThreeWayCrossover: Three-way crossover for woofer/midrange/tweeter
    FilteredWaveform: Waveform with applied filter
    DelayedWaveform: Waveform with time delay for driver alignment
    ParametricEQ: Multi-band parametric equalizer
    EQWaveform: Waveform with EQ applied
    EQBand: Single EQ band configuration

Example:
    >>> from strata_fdtd import AudioFileWaveform, Crossover, DelayedWaveform
    >>> from strata_fdtd.dsp import ParametricEQ
    >>>
    >>> # Load audio
    >>> audio = AudioFileWaveform("song.wav", duration=0.5)
    >>>
    >>> # Create 2kHz crossover
    >>> xover = Crossover(frequency=2000, order=4, type='linkwitz-riley')
    >>>
    >>> # Time-align tweeter (2mm ahead of woofer)
    >>> tweeter_waveform = DelayedWaveform(
    ...     source=xover.highpass(audio),
    ...     delay_seconds=2e-3 / 343,
    ... )
    >>>
    >>> # Assign to drivers
    >>> enc.add_driver(..., name="tweeter", waveform=tweeter_waveform)
    >>> enc.add_driver(..., name="woofer", waveform=xover.lowpass(audio))
"""

from strata_fdtd.dsp.crossover import (
    Crossover,
    FilteredWaveform,
    ThreeWayCrossover,
    WaveformProtocol,
)
from strata_fdtd.dsp.delay import DelayedWaveform
from strata_fdtd.dsp.eq import (
    EQBand,
    EQWaveform,
    ParametricEQ,
    create_baffle_step_compensation,
)

__all__ = [
    # Crossover filters
    "Crossover",
    "ThreeWayCrossover",
    "FilteredWaveform",
    "WaveformProtocol",
    # Delay
    "DelayedWaveform",
    # EQ
    "ParametricEQ",
    "EQWaveform",
    "EQBand",
    "create_baffle_step_compensation",
]
