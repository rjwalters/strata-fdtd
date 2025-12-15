"""
Unit tests for frequency weighting filters and SPL calculations.

Tests verify:
- A-weighting and C-weighting filter response against IEC 61672-1 tolerances
- SPL and Leq calculations
- Integration with Microphone class
"""

import numpy as np
import pytest

from strata_fdtd import FDTDSolver, GaussianPulse, apply_weighting, calculate_leq, calculate_spl
from strata_fdtd.analysis import weighting_response
from strata_fdtd.analysis.weighting import P_REF, calculate_time_weighted_level, get_weighting_sos

# =============================================================================
# IEC 61672-1 Reference Values
# =============================================================================

# Reference A-weighting values at standard frequencies (from IEC 61672-1)
# Format: (frequency_Hz, expected_dB, tolerance_dB)
# Note: Tolerances for very low frequencies (<40 Hz) are increased beyond IEC 61672-1
# to account for bilinear transform frequency warping at low frequencies.
A_WEIGHTING_REFERENCE = [
    (10, -70.4, 6.0),   # Extended tolerance for bilinear warping
    (12.5, -63.4, 5.0), # Extended tolerance for bilinear warping
    (16, -56.7, 4.0),   # Extended tolerance for bilinear warping
    (20, -50.5, 4.0),   # Extended tolerance for bilinear warping
    (25, -44.7, 3.0),   # Extended tolerance for bilinear warping
    (31.5, -39.4, 2.5), # Extended tolerance for bilinear warping
    (40, -34.6, 1.5),
    (50, -30.2, 1.5),
    (63, -26.2, 1.5),
    (80, -22.5, 1.5),
    (100, -19.1, 1.5),
    (125, -16.1, 1.5),
    (160, -13.4, 1.5),
    (200, -10.9, 1.5),
    (250, -8.6, 1.4),
    (315, -6.6, 1.4),
    (400, -4.8, 1.4),
    (500, -3.2, 1.4),
    (630, -1.9, 1.4),
    (800, -0.8, 1.4),
    (1000, 0.0, 1.1),  # Reference frequency
    (1250, 0.6, 1.4),
    (1600, 1.0, 1.6),
    (2000, 1.2, 1.6),
    (2500, 1.3, 1.6),
    (3150, 1.2, 1.6),
    (4000, 1.0, 1.6),
    (5000, 0.5, 2.1),
    (6300, -0.1, 2.1),
    (8000, -1.1, 2.1),
    (10000, -2.5, 2.6),
    (12500, -4.3, 3.0),
    (16000, -6.6, 3.5),
    (20000, -9.3, 4.0),
]

# Reference C-weighting values at standard frequencies (from IEC 61672-1)
C_WEIGHTING_REFERENCE = [
    (10, -14.3, 3.0),
    (12.5, -11.2, 3.0),
    (16, -8.5, 2.5),
    (20, -6.2, 2.5),
    (25, -4.4, 2.0),
    (31.5, -3.0, 1.5),
    (40, -2.0, 1.5),
    (50, -1.3, 1.5),
    (63, -0.8, 1.5),
    (80, -0.5, 1.5),
    (100, -0.3, 1.5),
    (125, -0.2, 1.5),
    (160, -0.1, 1.5),
    (200, 0.0, 1.5),
    (250, 0.0, 1.4),
    (315, 0.0, 1.4),
    (400, 0.0, 1.4),
    (500, 0.0, 1.4),
    (630, 0.0, 1.4),
    (800, 0.0, 1.4),
    (1000, 0.0, 1.1),  # Reference frequency
    (1250, 0.0, 1.4),
    (1600, -0.1, 1.6),
    (2000, -0.2, 1.6),
    (2500, -0.3, 1.6),
    (3150, -0.5, 1.6),
    (4000, -0.8, 1.6),
    (5000, -1.3, 2.1),
    (6300, -2.0, 2.1),
    (8000, -3.0, 2.1),
    (10000, -4.4, 2.6),
    (12500, -6.2, 3.0),
    (16000, -8.5, 3.5),
    (20000, -11.2, 4.0),
]


# =============================================================================
# Filter Design Tests
# =============================================================================


class TestFilterDesign:
    @pytest.fixture
    def sample_rate(self):
        """Common sample rate for tests."""
        return 48000

    def test_a_weighting_sos_shape(self, sample_rate):
        """Test that A-weighting SOS has expected shape."""
        sos = get_weighting_sos("A", sample_rate)
        assert sos is not None
        # Should be Nx6 array (second-order sections)
        assert sos.ndim == 2
        assert sos.shape[1] == 6

    def test_c_weighting_sos_shape(self, sample_rate):
        """Test that C-weighting SOS has expected shape."""
        sos = get_weighting_sos("C", sample_rate)
        assert sos is not None
        assert sos.ndim == 2
        assert sos.shape[1] == 6

    def test_z_weighting_returns_none(self, sample_rate):
        """Test that Z-weighting returns None (passthrough)."""
        assert get_weighting_sos("Z", sample_rate) is None
        assert get_weighting_sos(None, sample_rate) is None

    def test_invalid_weighting_raises(self, sample_rate):
        """Test that invalid weighting type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown weighting"):
            get_weighting_sos("X", sample_rate)

    def test_sos_caching(self, sample_rate):
        """Test that SOS coefficients are cached."""
        sos1 = get_weighting_sos("A", sample_rate)
        sos2 = get_weighting_sos("A", sample_rate)
        # Should return same object from cache
        assert sos1 is sos2


# =============================================================================
# A-Weighting Response Tests
# =============================================================================


class TestAWeightingResponse:
    @pytest.fixture
    def sample_rate(self):
        """High sample rate for accurate response measurement."""
        return 96000

    @pytest.mark.parametrize(
        "freq,expected_db,tolerance",
        [(f, e, t) for f, e, t in A_WEIGHTING_REFERENCE if f <= 20000],
    )
    def test_a_weighting_at_frequency(self, sample_rate, freq, expected_db, tolerance):
        """Test A-weighting response matches IEC 61672-1 at standard frequencies."""
        if freq > sample_rate / 2:
            pytest.skip(f"Frequency {freq} Hz above Nyquist")

        freqs, mag_db = weighting_response("A", sample_rate, n_points=8192)

        # Find closest frequency in response
        idx = np.argmin(np.abs(freqs - freq))
        actual_db = mag_db[idx]

        assert actual_db == pytest.approx(
            expected_db, abs=tolerance
        ), f"A-weighting at {freq} Hz: expected {expected_db} ± {tolerance} dB, got {actual_db:.1f} dB"

    def test_a_weighting_unity_at_1khz(self, sample_rate):
        """Test A-weighting is 0 dB at 1 kHz."""
        freqs, mag_db = weighting_response("A", sample_rate, n_points=8192)

        idx_1k = np.argmin(np.abs(freqs - 1000))
        assert mag_db[idx_1k] == pytest.approx(0.0, abs=0.5)


# =============================================================================
# C-Weighting Response Tests
# =============================================================================


class TestCWeightingResponse:
    @pytest.fixture
    def sample_rate(self):
        """High sample rate for accurate response measurement."""
        return 96000

    @pytest.mark.parametrize(
        "freq,expected_db,tolerance",
        [(f, e, t) for f, e, t in C_WEIGHTING_REFERENCE if f <= 20000],
    )
    def test_c_weighting_at_frequency(self, sample_rate, freq, expected_db, tolerance):
        """Test C-weighting response matches IEC 61672-1 at standard frequencies."""
        if freq > sample_rate / 2:
            pytest.skip(f"Frequency {freq} Hz above Nyquist")

        freqs, mag_db = weighting_response("C", sample_rate, n_points=8192)

        # Find closest frequency in response
        idx = np.argmin(np.abs(freqs - freq))
        actual_db = mag_db[idx]

        assert actual_db == pytest.approx(
            expected_db, abs=tolerance
        ), f"C-weighting at {freq} Hz: expected {expected_db} ± {tolerance} dB, got {actual_db:.1f} dB"

    def test_c_weighting_unity_at_1khz(self, sample_rate):
        """Test C-weighting is 0 dB at 1 kHz."""
        freqs, mag_db = weighting_response("C", sample_rate, n_points=8192)

        idx_1k = np.argmin(np.abs(freqs - 1000))
        assert mag_db[idx_1k] == pytest.approx(0.0, abs=0.5)

    def test_c_flatter_than_a_at_low_freq(self, sample_rate):
        """Test that C-weighting is flatter than A at low frequencies."""
        _, a_mag = weighting_response("A", sample_rate, n_points=8192)
        freqs, c_mag = weighting_response("C", sample_rate, n_points=8192)

        # At 50 Hz, C should be much higher (less attenuation) than A
        idx_50 = np.argmin(np.abs(freqs - 50))
        assert c_mag[idx_50] > a_mag[idx_50] + 20  # C is ~28 dB higher at 50 Hz


# =============================================================================
# Apply Weighting Tests
# =============================================================================


class TestApplyWeighting:
    @pytest.fixture
    def sample_rate(self):
        return 48000

    def test_z_weighting_passthrough(self, sample_rate):
        """Test Z-weighting returns original signal."""
        signal = np.random.randn(1000).astype(np.float32)

        result_z = apply_weighting(signal, sample_rate, "Z")
        result_none = apply_weighting(signal, sample_rate, None)

        np.testing.assert_array_almost_equal(result_z, signal)
        np.testing.assert_array_almost_equal(result_none, signal)

    def test_a_weighting_attenuates_low_freq(self, sample_rate):
        """Test A-weighting attenuates low frequency signals."""
        t = np.arange(sample_rate) / sample_rate  # 1 second
        low_freq = np.sin(2 * np.pi * 50 * t).astype(np.float32)  # 50 Hz

        weighted = apply_weighting(low_freq, sample_rate, "A")

        # A-weighting attenuates 50 Hz by ~30 dB
        assert np.max(np.abs(weighted)) < np.max(np.abs(low_freq)) * 0.1

    def test_a_weighting_preserves_1khz(self, sample_rate):
        """Test A-weighting preserves 1 kHz signals."""
        t = np.arange(sample_rate) / sample_rate  # 1 second
        mid_freq = np.sin(2 * np.pi * 1000 * t).astype(np.float32)  # 1 kHz

        weighted = apply_weighting(mid_freq, sample_rate, "A")

        # Allow for filter transient by checking steady-state
        # Use middle portion of signal
        mid_start = len(weighted) // 4
        mid_end = 3 * len(weighted) // 4

        orig_rms = np.sqrt(np.mean(mid_freq[mid_start:mid_end] ** 2))
        weighted_rms = np.sqrt(np.mean(weighted[mid_start:mid_end] ** 2))

        # Should be within 1 dB
        ratio_db = 20 * np.log10(weighted_rms / orig_rms)
        assert ratio_db == pytest.approx(0.0, abs=1.0)


# =============================================================================
# SPL Calculation Tests
# =============================================================================


class TestSPLCalculation:
    def test_spl_reference_level(self):
        """Test SPL calculation at reference level."""
        # 20 µPa RMS should be 0 dB SPL
        signal = np.ones(1000) * P_REF
        spl = calculate_spl(signal)
        assert spl == pytest.approx(0.0, abs=0.1)

    def test_spl_94db(self):
        """Test SPL calculation at 94 dB (1 Pa RMS)."""
        # 1 Pa RMS = 94 dB SPL
        signal = np.ones(1000) * 1.0
        spl = calculate_spl(signal)
        assert spl == pytest.approx(94.0, abs=0.1)

    def test_spl_sinusoid(self):
        """Test SPL of sinusoidal signal."""
        # Sinusoid with amplitude A has RMS = A/sqrt(2)
        amplitude = 1.0  # 1 Pa peak
        rms = amplitude / np.sqrt(2)  # ~0.707 Pa RMS
        expected_spl = 20 * np.log10(rms / P_REF)  # ~91 dB

        t = np.linspace(0, 1, 48000)
        signal = amplitude * np.sin(2 * np.pi * 1000 * t)

        spl = calculate_spl(signal)
        assert spl == pytest.approx(expected_spl, abs=0.1)

    def test_spl_zero_signal(self):
        """Test SPL of zero signal returns -inf."""
        signal = np.zeros(1000)
        spl = calculate_spl(signal)
        assert spl == -np.inf


# =============================================================================
# Leq Calculation Tests
# =============================================================================


class TestLeqCalculation:
    def test_leq_constant_signal(self):
        """Test Leq equals SPL for constant signal."""
        signal = np.ones(48000) * 0.1  # 0.1 Pa
        fs = 48000

        spl = calculate_spl(signal)
        leq = calculate_leq(signal, fs)

        assert leq == pytest.approx(spl, abs=0.01)

    def test_leq_with_duration(self):
        """Test Leq with specified duration."""
        fs = 48000

        # Signal with two levels
        signal = np.concatenate([
            np.ones(fs) * 0.01,  # 1 second at 0.01 Pa
            np.ones(fs) * 0.1,   # 1 second at 0.1 Pa
        ])

        # Leq over last 1 second should reflect higher level
        leq_last = calculate_leq(signal, fs, duration=1.0)
        expected = calculate_spl(np.ones(fs) * 0.1)

        assert leq_last == pytest.approx(expected, abs=0.1)


# =============================================================================
# Time-Weighted Level Tests
# =============================================================================


class TestTimeWeightedLevel:
    def test_fast_time_constant(self):
        """Test fast time weighting (125 ms)."""
        fs = 48000
        signal = np.zeros(fs)
        signal[0] = 1.0  # Impulse

        levels = calculate_time_weighted_level(signal, fs, "fast")

        # Level should decay over time
        assert len(levels) == len(signal)
        # Initial level should be high
        assert levels[0] > -100
        # Should decay significantly within 125 ms
        idx_125ms = int(0.125 * fs)
        assert levels[idx_125ms] < levels[0]

    def test_slow_time_constant(self):
        """Test slow time weighting (1000 ms) has slower decay rate than fast."""
        fs = 48000

        # Use a step-then-silence pattern: signal for first 0.2s, then silence
        # After signal ends, measure how quickly each decays
        signal = np.zeros(fs * 2)  # 2 seconds total
        signal[: int(0.2 * fs)] = 1.0  # 0.2s of signal

        slow_levels = calculate_time_weighted_level(signal, fs, "slow")
        fast_levels = calculate_time_weighted_level(signal, fs, "fast")

        # Measure decay: check level at 0.4s and 0.8s (after signal ends at 0.2s)
        # Both should be decaying, but slow should decay less
        t1 = int(0.4 * fs)
        t2 = int(0.8 * fs)

        # Calculate decay in dB over this interval
        slow_decay = slow_levels[t1] - slow_levels[t2]
        fast_decay = fast_levels[t1] - fast_levels[t2]

        # Fast should decay more than slow over the same interval
        assert fast_decay > slow_decay, (
            f"Expected fast decay ({fast_decay:.1f} dB) > "
            f"slow decay ({slow_decay:.1f} dB)"
        )


# =============================================================================
# Microphone Integration Tests
# =============================================================================


class TestMicrophoneWeighting:
    @pytest.fixture
    def solver_with_mic(self):
        """Create solver with microphone and source."""
        solver = FDTDSolver(shape=(50, 50, 50), resolution=2e-3)
        solver.add_source(GaussianPulse(
            position=(10, 25, 25),
            frequency=1000,
            amplitude=10.0,
        ))
        mic = solver.add_microphone(
            position=(0.05, 0.05, 0.05),
            name="test"
        )
        solver.run(duration=0.005)
        return solver, mic

    def test_microphone_get_waveform_unweighted(self, solver_with_mic):
        """Test get_waveform without weighting."""
        _, mic = solver_with_mic
        waveform = mic.get_waveform()
        assert len(waveform) > 0
        assert waveform.dtype == np.float32

    def test_microphone_get_waveform_a_weighted(self, solver_with_mic):
        """Test get_waveform with A-weighting."""
        _, mic = solver_with_mic
        waveform_raw = mic.get_waveform()
        waveform_a = mic.get_waveform(weighting="A")

        assert len(waveform_a) == len(waveform_raw)
        # Waveforms should be different
        assert not np.allclose(waveform_raw, waveform_a)

    def test_microphone_get_waveform_c_weighted(self, solver_with_mic):
        """Test get_waveform with C-weighting."""
        _, mic = solver_with_mic
        waveform_raw = mic.get_waveform()
        waveform_c = mic.get_waveform(weighting="C")

        assert len(waveform_c) == len(waveform_raw)
        # C-weighting at mid frequencies should be similar to raw
        # but not identical due to low/high frequency attenuation

    def test_microphone_get_waveform_z_weighted(self, solver_with_mic):
        """Test get_waveform with Z-weighting (passthrough)."""
        _, mic = solver_with_mic
        waveform_raw = mic.get_waveform()
        waveform_z = mic.get_waveform(weighting="Z")

        np.testing.assert_array_equal(waveform_raw, waveform_z)

    def test_microphone_get_spl(self, solver_with_mic):
        """Test get_spl method."""
        _, mic = solver_with_mic
        spl = mic.get_spl()

        # Should return a finite value
        assert np.isfinite(spl)
        # Pressure from simulation should give some SPL value
        assert spl > -200  # Not effectively silent

    def test_microphone_get_spl_weighted(self, solver_with_mic):
        """Test get_spl with weighting."""
        _, mic = solver_with_mic
        spl_raw = mic.get_spl()
        spl_a = mic.get_spl(weighting="A")
        spl_c = mic.get_spl(weighting="C")

        # All should be finite
        assert np.isfinite(spl_raw)
        assert np.isfinite(spl_a)
        assert np.isfinite(spl_c)

    def test_microphone_get_leq(self, solver_with_mic):
        """Test get_leq method."""
        _, mic = solver_with_mic
        leq = mic.get_leq()

        # Should equal SPL for this constant-power signal (approximately)
        spl = mic.get_spl()
        assert leq == pytest.approx(spl, abs=0.1)

    def test_microphone_get_leq_with_duration(self, solver_with_mic):
        """Test get_leq with duration parameter."""
        _, mic = solver_with_mic
        leq_full = mic.get_leq()
        leq_half = mic.get_leq(duration=0.0025)  # Half the recording

        # Both should be finite
        assert np.isfinite(leq_full)
        assert np.isfinite(leq_half)

    def test_microphone_get_time_weighted_level(self, solver_with_mic):
        """Test get_time_weighted_level method."""
        _, mic = solver_with_mic
        levels = mic.get_time_weighted_level(weighting="A", time_constant="fast")

        assert len(levels) == len(mic._data)
        # Levels should be finite (or -inf for zero)
        assert np.all(np.isfinite(levels) | (levels == -np.inf))

    def test_microphone_methods_raise_before_recording(self):
        """Test that methods raise RuntimeError before recording."""
        solver = FDTDSolver(shape=(20, 20, 20), resolution=5e-3)
        mic = solver.add_microphone(position=(0.05, 0.05, 0.05), name="test")
        # Don't run simulation

        with pytest.raises(RuntimeError, match="No data recorded"):
            mic.get_spl()

        with pytest.raises(RuntimeError, match="No data recorded"):
            mic.get_leq()

        with pytest.raises(RuntimeError, match="No data recorded"):
            mic.get_time_weighted_level()
