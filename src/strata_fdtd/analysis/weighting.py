"""
Frequency weighting filters for acoustic measurements.

Implements A-weighting, C-weighting, and Z-weighting per IEC 61672-1.
Filters are designed using the bilinear transform from analog prototypes.

Typical usage:
    >>> waveform = mic.get_waveform()
    >>> sample_rate = mic.get_sample_rate()
    >>> weighted = apply_weighting(waveform, sample_rate, "A")
    >>> spl_dba = calculate_spl(weighted)
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal

# IEC 61672-1 frequency constants (Hz)
_F1 = 20.598997
_F2 = 107.65265
_F3 = 737.86223
_F4 = 12194.217

# Angular frequencies
_W1 = 2 * np.pi * _F1
_W2 = 2 * np.pi * _F2
_W3 = 2 * np.pi * _F3
_W4 = 2 * np.pi * _F4

# Reference sound pressure level (20 µPa)
P_REF = 2e-5

WeightingType = Literal["A", "C", "Z", None]


def _design_a_weighting_sos(fs: float) -> NDArray[np.floating]:
    """Design A-weighting filter as second-order sections.

    The A-weighting transfer function (analog):

        H_A(s) = k_A * s^4 / [(s + ω1)^2 * (s + ω2) * (s + ω3) * (s + ω4)^2]

    This approximates the equal-loudness contour at ~40 phon.

    Args:
        fs: Sample rate in Hz

    Returns:
        Second-order section coefficients (N x 6 array)
    """
    # Analog zeros and poles
    zeros = [0, 0, 0, 0]  # 4 zeros at origin (s^4 numerator)
    poles = [
        -_W1, -_W1,  # Double pole at f1
        -_W2,        # Single pole at f2
        -_W3,        # Single pole at f3
        -_W4, -_W4,  # Double pole at f4
    ]

    # Gain to achieve 0 dB at 1 kHz
    # k = ω4² * ω1² / (ω4² + (2πf_ref)²) ... etc
    # For A-weighting, we normalize so response = 0 dB at 1 kHz
    f_ref = 1000.0
    w_ref = 2 * np.pi * f_ref

    # Compute analog gain (magnitude at 1 kHz should be 1)
    # |H(jω)| = k * ω^4 / |terms|
    num_mag = w_ref**4
    denom_terms = [
        np.sqrt(w_ref**2 + _W1**2),  # (s + ω1) for s=jω
        np.sqrt(w_ref**2 + _W1**2),  # (s + ω1) again
        np.sqrt(w_ref**2 + _W2**2),  # (s + ω2)
        np.sqrt(w_ref**2 + _W3**2),  # (s + ω3)
        np.sqrt(w_ref**2 + _W4**2),  # (s + ω4)
        np.sqrt(w_ref**2 + _W4**2),  # (s + ω4) again
    ]
    denom_mag = np.prod(denom_terms)
    k = denom_mag / num_mag

    # Convert to digital filter using bilinear transform
    z, p, k_digital = signal.bilinear_zpk(zeros, poles, k, fs)

    # Convert to second-order sections for numerical stability
    sos = signal.zpk2sos(z, p, k_digital)

    return sos


def _design_c_weighting_sos(fs: float) -> NDArray[np.floating]:
    """Design C-weighting filter as second-order sections.

    The C-weighting transfer function (analog):

        H_C(s) = k_C * s^2 / [(s + ω1)^2 * (s + ω4)^2]

    C-weighting is flatter than A-weighting, used for high-level sounds.

    Args:
        fs: Sample rate in Hz

    Returns:
        Second-order section coefficients (N x 6 array)
    """
    # Analog zeros and poles
    zeros = [0, 0]  # 2 zeros at origin (s^2 numerator)
    poles = [
        -_W1, -_W1,  # Double pole at f1
        -_W4, -_W4,  # Double pole at f4
    ]

    # Gain to achieve 0 dB at 1 kHz
    f_ref = 1000.0
    w_ref = 2 * np.pi * f_ref

    num_mag = w_ref**2
    denom_terms = [
        np.sqrt(w_ref**2 + _W1**2),
        np.sqrt(w_ref**2 + _W1**2),
        np.sqrt(w_ref**2 + _W4**2),
        np.sqrt(w_ref**2 + _W4**2),
    ]
    denom_mag = np.prod(denom_terms)
    k = denom_mag / num_mag

    # Convert to digital filter using bilinear transform
    z, p, k_digital = signal.bilinear_zpk(zeros, poles, k, fs)

    # Convert to second-order sections for numerical stability
    sos = signal.zpk2sos(z, p, k_digital)

    return sos


# Cache for filter coefficients (keyed by sample rate)
_sos_cache: dict[tuple[WeightingType, float], NDArray[np.floating]] = {}


def get_weighting_sos(
    weighting: WeightingType,
    fs: float,
) -> NDArray[np.floating] | None:
    """Get second-order section filter coefficients for frequency weighting.

    Args:
        weighting: Weighting type ("A", "C", "Z", or None)
        fs: Sample rate in Hz

    Returns:
        SOS coefficients array, or None for Z-weighting/passthrough

    Raises:
        ValueError: If weighting type is not recognized
    """
    if weighting in ("Z", None):
        return None

    cache_key = (weighting, fs)
    if cache_key in _sos_cache:
        return _sos_cache[cache_key]

    if weighting == "A":
        sos = _design_a_weighting_sos(fs)
    elif weighting == "C":
        sos = _design_c_weighting_sos(fs)
    else:
        raise ValueError(
            f"Unknown weighting type: {weighting!r}. "
            f"Valid options: 'A', 'C', 'Z', or None"
        )

    _sos_cache[cache_key] = sos
    return sos


def apply_weighting(
    waveform: NDArray[np.floating],
    fs: float,
    weighting: WeightingType,
) -> NDArray[np.floating]:
    """Apply frequency weighting to a waveform.

    Args:
        waveform: Input pressure waveform
        fs: Sample rate in Hz
        weighting: Weighting type ("A", "C", "Z", or None)

    Returns:
        Weighted waveform (same length as input)
    """
    sos = get_weighting_sos(weighting, fs)

    if sos is None:
        return waveform.astype(np.float64)

    # Apply filter using SOS for numerical stability
    return signal.sosfilt(sos, waveform).astype(np.float64)


def calculate_spl(
    waveform: NDArray[np.floating],
    p_ref: float = P_REF,
) -> float:
    """Calculate sound pressure level (SPL) in dB.

    SPL = 20 * log10(p_rms / p_ref)

    Args:
        waveform: Pressure waveform in Pa
        p_ref: Reference pressure (default: 20 µPa for air)

    Returns:
        SPL in dB (or dBA/dBC if waveform was pre-weighted)
    """
    p_rms = np.sqrt(np.mean(waveform**2))

    if p_rms == 0:
        return -np.inf

    return 20 * np.log10(p_rms / p_ref)


def calculate_leq(
    waveform: NDArray[np.floating],
    fs: float,
    duration: float | None = None,
    p_ref: float = P_REF,
) -> float:
    """Calculate equivalent continuous sound level (Leq).

    Leq = 10 * log10((1/T) * integral(p(t)² dt) / p_ref²)

    For discrete signals, this simplifies to the RMS level.

    Args:
        waveform: Pressure waveform in Pa
        fs: Sample rate in Hz
        duration: Integration time in seconds (default: use full waveform)
        p_ref: Reference pressure (default: 20 µPa for air)

    Returns:
        Leq in dB
    """
    if duration is not None:
        n_samples = int(duration * fs)
        if n_samples < len(waveform):
            waveform = waveform[-n_samples:]  # Use most recent samples

    return calculate_spl(waveform, p_ref)


def calculate_time_weighted_level(
    waveform: NDArray[np.floating],
    fs: float,
    time_constant: Literal["fast", "slow", "impulse"] = "fast",
    p_ref: float = P_REF,
) -> NDArray[np.floating]:
    """Calculate time-weighted sound levels.

    Applies exponential averaging with standard time constants:
    - Fast: 125 ms
    - Slow: 1000 ms
    - Impulse: 35 ms attack, 1500 ms decay (approximated as 35 ms)

    Args:
        waveform: Pressure waveform in Pa
        fs: Sample rate in Hz
        time_constant: "fast", "slow", or "impulse"
        p_ref: Reference pressure (default: 20 µPa for air)

    Returns:
        Array of instantaneous SPL values in dB
    """
    tau_map = {
        "fast": 0.125,
        "slow": 1.0,
        "impulse": 0.035,  # Attack time constant
    }

    tau = tau_map[time_constant]

    # Exponential averaging filter
    # y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    # where alpha = 1 - exp(-1/(fs*tau))
    alpha = 1 - np.exp(-1 / (fs * tau))

    # Square the waveform (for power averaging)
    p_squared = waveform**2

    # Apply single-pole IIR filter for exponential averaging
    b = [alpha]
    a = [1, -(1 - alpha)]
    p_avg = signal.lfilter(b, a, p_squared)

    # Convert to dB
    # Handle zeros by clipping to small positive value
    p_avg = np.maximum(p_avg, 1e-20)
    spl = 10 * np.log10(p_avg / p_ref**2)

    return spl


def weighting_response(
    weighting: WeightingType,
    fs: float,
    n_points: int = 1024,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the frequency response of a weighting filter.

    Useful for verification against IEC 61672-1 tables.

    Args:
        weighting: Weighting type ("A", "C")
        fs: Sample rate in Hz
        n_points: Number of frequency points

    Returns:
        Tuple of (frequencies, magnitude_dB)
    """
    sos = get_weighting_sos(weighting, fs)

    if sos is None:
        freqs = np.linspace(0, fs/2, n_points)
        return freqs, np.zeros(n_points)

    # Compute frequency response
    w, h = signal.sosfreqz(sos, worN=n_points, fs=fs)

    # Convert to dB
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-20))

    return w, mag_db
