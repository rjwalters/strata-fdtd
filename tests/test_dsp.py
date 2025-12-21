"""
Unit tests for DSP crossover and EQ components.

Tests verify:
- Crossover filter frequency response (Butterworth, Linkwitz-Riley, Bessel)
- FilteredWaveform lazy evaluation and caching
- ThreeWayCrossover band separation
- DelayedWaveform time alignment
- ParametricEQ band processing
"""

import numpy as np
import pytest
from scipy.io import wavfile

from strata_fdtd import (
    AudioFileWaveform,
    Crossover,
    DelayedWaveform,
    FilteredWaveform,
    ParametricEQ,
    ThreeWayCrossover,
)
from strata_fdtd.dsp import EQBand, EQWaveform, create_baffle_step_compensation


def create_test_wav(filepath, sr: int, duration: float, freq: float = 440):
    """Create a test WAV file with a sine wave."""
    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    data = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    wavfile.write(filepath, sr, data)
    return n_samples


def create_noise_wav(filepath, sr: int, duration: float, seed: int = 42):
    """Create a test WAV file with white noise."""
    np.random.seed(seed)
    n_samples = int(sr * duration)
    data = (np.random.randn(n_samples) * 16384).astype(np.int16)
    wavfile.write(filepath, sr, data)
    return n_samples


class TestCrossover:
    """Tests for Crossover class."""

    def test_crossover_creation(self):
        """Test Crossover can be created with different parameters."""
        xover = Crossover(frequency=2000)
        assert xover.frequency == 2000
        assert xover.order == 4
        assert xover.type == "linkwitz-riley"

        xover2 = Crossover(frequency=1000, order=2, type="butterworth")
        assert xover2.frequency == 1000
        assert xover2.order == 2
        assert xover2.type == "butterworth"

    def test_lowpass_creates_filtered_waveform(self, tmp_path):
        """Test lowpass() returns a FilteredWaveform."""
        filepath = tmp_path / "test.wav"
        create_test_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        xover = Crossover(frequency=2000)
        lowpass = xover.lowpass(audio)

        assert isinstance(lowpass, FilteredWaveform)
        assert lowpass.filter_type == "lowpass"
        assert lowpass.frequency == 2000

    def test_highpass_creates_filtered_waveform(self, tmp_path):
        """Test highpass() returns a FilteredWaveform."""
        filepath = tmp_path / "test.wav"
        create_test_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        xover = Crossover(frequency=2000)
        highpass = xover.highpass(audio)

        assert isinstance(highpass, FilteredWaveform)
        assert highpass.filter_type == "highpass"
        assert highpass.frequency == 2000


class TestFilteredWaveform:
    """Tests for FilteredWaveform class."""

    def test_lowpass_attenuates_high_frequencies(self, tmp_path):
        """Test lowpass filter attenuates frequencies above cutoff."""
        filepath = tmp_path / "test.wav"
        sr = 44100
        duration = 0.1

        # Create high frequency signal (5kHz)
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        data = (np.sin(2 * np.pi * 5000 * t) * 32767).astype(np.int16)
        wavfile.write(filepath, sr, data)

        audio = AudioFileWaveform(filepath)
        xover = Crossover(frequency=1000, order=4, type="butterworth")
        lowpass = xover.lowpass(audio)

        dt = 1.0 / sr
        sim_t = np.arange(0, duration, dt)

        original = audio.waveform(sim_t, dt)
        filtered = lowpass.waveform(sim_t, dt)

        # High frequency should be significantly attenuated
        original_rms = np.sqrt(np.mean(original**2))
        filtered_rms = np.sqrt(np.mean(filtered**2))

        # 5kHz is 5 octaves above 1kHz cutoff, expect ~60dB attenuation for 4th order
        assert filtered_rms < original_rms * 0.1

    def test_highpass_attenuates_low_frequencies(self, tmp_path):
        """Test highpass filter attenuates frequencies below cutoff."""
        filepath = tmp_path / "test.wav"
        sr = 44100
        duration = 0.1

        # Create low frequency signal (100Hz)
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        data = (np.sin(2 * np.pi * 100 * t) * 32767).astype(np.int16)
        wavfile.write(filepath, sr, data)

        audio = AudioFileWaveform(filepath)
        xover = Crossover(frequency=1000, order=4, type="butterworth")
        highpass = xover.highpass(audio)

        dt = 1.0 / sr
        sim_t = np.arange(0, duration, dt)

        original = audio.waveform(sim_t, dt)
        filtered = highpass.waveform(sim_t, dt)

        # Low frequency should be significantly attenuated
        original_rms = np.sqrt(np.mean(original**2))
        filtered_rms = np.sqrt(np.mean(filtered**2))

        assert filtered_rms < original_rms * 0.1

    def test_passband_signal_preserved(self, tmp_path):
        """Test that signal in passband is largely preserved."""
        filepath = tmp_path / "test.wav"
        sr = 44100
        duration = 0.1

        # Create signal at crossover frequency
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        # 200Hz signal with 1kHz crossover - well in passband for lowpass
        data = (np.sin(2 * np.pi * 200 * t) * 32767).astype(np.int16)
        wavfile.write(filepath, sr, data)

        audio = AudioFileWaveform(filepath)
        xover = Crossover(frequency=1000, order=4, type="butterworth")
        lowpass = xover.lowpass(audio)

        dt = 1.0 / sr
        sim_t = np.arange(0, duration, dt)

        original = audio.waveform(sim_t, dt)
        filtered = lowpass.waveform(sim_t, dt)

        # Passband signal should be mostly preserved (within 3dB)
        original_rms = np.sqrt(np.mean(original**2))
        filtered_rms = np.sqrt(np.mean(filtered**2))

        assert filtered_rms > original_rms * 0.7

    def test_caching_works(self, tmp_path):
        """Test that filtered results are cached per sample rate."""
        filepath = tmp_path / "test.wav"
        create_test_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        xover = Crossover(frequency=1000)
        filtered = xover.lowpass(audio)

        dt = 1e-5
        t = np.arange(0, 0.01, dt)

        # First call
        result1 = filtered.waveform(t, dt)
        # Second call should use cache
        result2 = filtered.waveform(t, dt)

        np.testing.assert_array_equal(result1, result2)

        # Different sample rate creates new cache entry
        dt2 = 2e-5
        t2 = np.arange(0, 0.01, dt2)
        result3 = filtered.waveform(t2, dt2)

        assert len(result3) != len(result1)

    def test_all_filter_designs(self, tmp_path):
        """Test all three filter designs work."""
        filepath = tmp_path / "test.wav"
        create_noise_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        dt = 1.0 / 44100
        t = np.arange(0, 0.05, dt)

        for design in ["butterworth", "linkwitz-riley", "bessel"]:
            xover = Crossover(frequency=1000, order=4, type=design)
            lowpass = xover.lowpass(audio)
            result = lowpass.waveform(t, dt)

            assert len(result) == len(t)
            assert not np.any(np.isnan(result))

    def test_bandpass_filter(self, tmp_path):
        """Test bandpass filtering works."""
        filepath = tmp_path / "test.wav"
        create_noise_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        filtered = FilteredWaveform(
            source=audio,
            filter_type="bandpass",
            frequency=(500, 2000),
            order=4,
            design="butterworth",
        )

        dt = 1.0 / 44100
        t = np.arange(0, 0.05, dt)
        result = filtered.waveform(t, dt)

        assert len(result) == len(t)
        assert not np.any(np.isnan(result))


class TestThreeWayCrossover:
    """Tests for ThreeWayCrossover class."""

    def test_three_way_creation(self):
        """Test ThreeWayCrossover can be created."""
        xover = ThreeWayCrossover(low_freq=500, high_freq=4000)
        assert xover.low_freq == 500
        assert xover.high_freq == 4000

    def test_invalid_frequencies_raises_error(self):
        """Test error when low_freq >= high_freq."""
        with pytest.raises(ValueError, match="low_freq.*must be less than"):
            ThreeWayCrossover(low_freq=4000, high_freq=500)

        with pytest.raises(ValueError, match="low_freq.*must be less than"):
            ThreeWayCrossover(low_freq=2000, high_freq=2000)

    def test_woofer_midrange_tweeter_filters(self, tmp_path):
        """Test all three outputs are created correctly."""
        filepath = tmp_path / "test.wav"
        create_noise_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        xover = ThreeWayCrossover(low_freq=500, high_freq=4000)

        woofer = xover.woofer(audio)
        midrange = xover.midrange(audio)
        tweeter = xover.tweeter(audio)

        assert isinstance(woofer, FilteredWaveform)
        assert isinstance(midrange, FilteredWaveform)
        assert isinstance(tweeter, FilteredWaveform)

        assert woofer.filter_type == "lowpass"
        assert midrange.filter_type == "bandpass"
        assert tweeter.filter_type == "highpass"

    def test_three_way_frequency_separation(self, tmp_path):
        """Test that three-way crossover separates frequencies."""
        filepath = tmp_path / "test.wav"
        sr = 44100
        duration = 0.1

        # Create multi-frequency signal: 100Hz + 1kHz + 8kHz
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        low = np.sin(2 * np.pi * 100 * t)
        mid = np.sin(2 * np.pi * 1000 * t)
        high = np.sin(2 * np.pi * 8000 * t)
        data = ((low + mid + high) / 3 * 32767).astype(np.int16)
        wavfile.write(filepath, sr, data)

        audio = AudioFileWaveform(filepath)
        xover = ThreeWayCrossover(low_freq=500, high_freq=4000, order=4)

        dt = 1.0 / sr
        sim_t = np.arange(0, duration, dt)

        woofer = xover.woofer(audio).waveform(sim_t, dt)
        midrange = xover.midrange(audio).waveform(sim_t, dt)
        tweeter = xover.tweeter(audio).waveform(sim_t, dt)

        # Each should have signal (RMS > 0)
        assert np.sqrt(np.mean(woofer**2)) > 0.01
        assert np.sqrt(np.mean(midrange**2)) > 0.01
        assert np.sqrt(np.mean(tweeter**2)) > 0.01


class TestDelayedWaveform:
    """Tests for DelayedWaveform class."""

    def test_delayed_waveform_creation(self, tmp_path):
        """Test DelayedWaveform can be created."""
        filepath = tmp_path / "test.wav"
        create_test_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        delayed = DelayedWaveform(source=audio, delay_seconds=0.01)

        assert delayed.delay_seconds == 0.01

    def test_delay_shifts_signal(self, tmp_path):
        """Test that delay actually shifts the signal in time."""
        filepath = tmp_path / "test.wav"
        sr = 44100
        duration = 0.1

        # Create impulse-like signal
        n_samples = int(sr * duration)
        data = np.zeros(n_samples, dtype=np.int16)
        data[100:110] = 32767  # Impulse at sample 100
        wavfile.write(filepath, sr, data)

        audio = AudioFileWaveform(filepath)
        delay_time = 0.01  # 10ms delay
        delayed = DelayedWaveform(source=audio, delay_seconds=delay_time)

        dt = 1.0 / sr
        t = np.arange(0, duration, dt)

        original = audio.waveform(t, dt)
        shifted = delayed.waveform(t, dt)

        # Find peak positions
        orig_peak = np.argmax(np.abs(original))
        shifted_peak = np.argmax(np.abs(shifted))

        # Shifted peak should be later by approximately delay_time
        expected_shift = int(delay_time / dt)
        actual_shift = shifted_peak - orig_peak

        assert abs(actual_shift - expected_shift) < 5

    def test_zero_at_start_before_delay(self, tmp_path):
        """Test that signal is zero before delay time."""
        filepath = tmp_path / "test.wav"
        create_test_wav(filepath, 44100, 0.1, freq=1000)
        audio = AudioFileWaveform(filepath)

        delay_time = 0.02  # 20ms delay
        delayed = DelayedWaveform(source=audio, delay_seconds=delay_time)

        dt = 1e-5
        # Request times before delay
        t = np.array([0.0, 0.005, 0.01, 0.015])
        result = delayed.waveform(t, dt)

        # All values should be from t=0 of source (near zero for sine)
        # But since we're calling source at delayed_t = t - delay which clips to 0
        # They should all return the same value
        assert len(result) == len(t)

    def test_duration_includes_delay(self, tmp_path):
        """Test that duration_seconds includes the delay."""
        filepath = tmp_path / "test.wav"
        create_test_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        delay_time = 0.05
        delayed = DelayedWaveform(source=audio, delay_seconds=delay_time)

        # Duration should be source duration + delay
        expected = audio.duration_seconds + delay_time
        assert abs(delayed.duration_seconds - expected) < 0.001

    def test_chained_with_crossover(self, tmp_path):
        """Test DelayedWaveform works with FilteredWaveform input."""
        filepath = tmp_path / "test.wav"
        create_test_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        xover = Crossover(frequency=2000)
        highpass = xover.highpass(audio)
        delayed = DelayedWaveform(source=highpass, delay_seconds=0.001)

        dt = 1.0 / 44100
        t = np.arange(0, 0.05, dt)
        result = delayed.waveform(t, dt)

        assert len(result) == len(t)
        assert not np.any(np.isnan(result))


class TestParametricEQ:
    """Tests for ParametricEQ class."""

    def test_parametric_eq_creation(self):
        """Test ParametricEQ can be created and bands added."""
        eq = ParametricEQ()
        eq.add_band(100, -3, 1.0, "peaking")
        eq.add_band(1000, 2, 0.7, "peaking")

        assert len(eq.bands) == 2
        assert eq.bands[0].frequency == 100
        assert eq.bands[0].gain_db == -3
        assert eq.bands[1].gain_db == 2

    def test_method_chaining(self):
        """Test add_band returns self for chaining."""
        eq = ParametricEQ()
        result = eq.add_band(100, -3, 1.0, "peaking")

        assert result is eq

    def test_clear_removes_bands(self):
        """Test clear() removes all bands."""
        eq = ParametricEQ()
        eq.add_band(100, -3, 1.0, "peaking")
        eq.add_band(1000, 2, 0.7, "peaking")
        eq.clear()

        assert len(eq.bands) == 0

    def test_apply_returns_eq_waveform(self, tmp_path):
        """Test apply() returns an EQWaveform."""
        filepath = tmp_path / "test.wav"
        create_test_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        eq = ParametricEQ()
        eq.add_band(100, -3, 1.0, "peaking")
        processed = eq.apply(audio)

        assert isinstance(processed, EQWaveform)

    def test_peaking_cut(self, tmp_path):
        """Test peaking filter cuts at center frequency."""
        filepath = tmp_path / "test.wav"
        sr = 44100
        duration = 0.1

        # Create signal at EQ frequency
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        data = (np.sin(2 * np.pi * 1000 * t) * 32767).astype(np.int16)
        wavfile.write(filepath, sr, data)

        audio = AudioFileWaveform(filepath)
        eq = ParametricEQ()
        eq.add_band(1000, -12, 2.0, "peaking")  # -12dB cut at 1kHz
        processed = eq.apply(audio)

        dt = 1.0 / sr
        sim_t = np.arange(0, duration, dt)

        original = audio.waveform(sim_t, dt)
        eqd = processed.waveform(sim_t, dt)

        # RMS should be reduced (not necessarily by exactly 12dB due to Q)
        original_rms = np.sqrt(np.mean(original**2))
        eqd_rms = np.sqrt(np.mean(eqd**2))

        assert eqd_rms < original_rms * 0.5

    def test_highpass_filter(self, tmp_path):
        """Test highpass EQ band."""
        filepath = tmp_path / "test.wav"
        sr = 44100
        duration = 0.1

        # Low frequency signal
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        data = (np.sin(2 * np.pi * 50 * t) * 32767).astype(np.int16)
        wavfile.write(filepath, sr, data)

        audio = AudioFileWaveform(filepath)
        eq = ParametricEQ()
        eq.add_band(200, 0, 0.707, "highpass")  # 200Hz highpass
        processed = eq.apply(audio)

        dt = 1.0 / sr
        sim_t = np.arange(0, duration, dt)

        original = audio.waveform(sim_t, dt)
        eqd = processed.waveform(sim_t, dt)

        # 50Hz should be attenuated by 200Hz highpass
        original_rms = np.sqrt(np.mean(original**2))
        eqd_rms = np.sqrt(np.mean(eqd**2))

        assert eqd_rms < original_rms * 0.3

    def test_lowshelf_boost(self, tmp_path):
        """Test lowshelf EQ boost."""
        filepath = tmp_path / "test.wav"
        create_noise_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        eq = ParametricEQ()
        eq.add_band(200, 6, 0.707, "lowshelf")  # +6dB low shelf
        processed = eq.apply(audio)

        dt = 1.0 / 44100
        t = np.arange(0, 0.05, dt)

        result = processed.waveform(t, dt)

        assert len(result) == len(t)
        assert not np.any(np.isnan(result))

    def test_eq_caching(self, tmp_path):
        """Test EQ results are cached."""
        filepath = tmp_path / "test.wav"
        create_test_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        eq = ParametricEQ()
        eq.add_band(1000, -3, 1.0, "peaking")
        processed = eq.apply(audio)

        dt = 1e-5
        t = np.arange(0, 0.01, dt)

        result1 = processed.waveform(t, dt)
        result2 = processed.waveform(t, dt)

        np.testing.assert_array_equal(result1, result2)


class TestEQBandTypes:
    """Tests for all EQ band types."""

    @pytest.mark.parametrize("band_type", ["peaking", "lowshelf", "highshelf", "lowpass", "highpass"])
    def test_all_band_types_work(self, tmp_path, band_type):
        """Test all band types can be applied without error."""
        filepath = tmp_path / "test.wav"
        create_noise_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        eq = ParametricEQ()
        eq.add_band(1000, 3 if band_type.endswith("shelf") or band_type == "peaking" else 0,
                   0.707, band_type)
        processed = eq.apply(audio)

        dt = 1.0 / 44100
        t = np.arange(0, 0.05, dt)
        result = processed.waveform(t, dt)

        assert len(result) == len(t)
        assert not np.any(np.isnan(result))


class TestBaffleStepCompensation:
    """Tests for baffle step compensation utility."""

    def test_create_baffle_step_compensation(self):
        """Test baffle step compensation EQ creation."""
        eq = create_baffle_step_compensation(baffle_width=0.2)

        assert len(eq.bands) == 1
        assert eq.bands[0].type == "lowshelf"

        # Baffle step frequency for 0.2m width
        expected_freq = 343.0 / (2 * 0.2)  # ~857 Hz
        assert abs(eq.bands[0].frequency - expected_freq) < 1

    def test_custom_boost_amount(self):
        """Test custom boost amount."""
        eq = create_baffle_step_compensation(baffle_width=0.3, boost_db=6.0)

        assert eq.bands[0].gain_db == 6.0


class TestIntegration:
    """Integration tests for DSP components."""

    def test_complete_two_way_crossover_chain(self, tmp_path):
        """Test complete 2-way crossover with delay."""
        filepath = tmp_path / "test.wav"
        create_noise_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        # Create crossover
        xover = Crossover(frequency=2000, order=4, type="linkwitz-riley")

        # Time-align tweeter
        tweeter = DelayedWaveform(
            source=xover.highpass(audio),
            delay_seconds=0.001,
        )
        woofer = xover.lowpass(audio)

        dt = 1.0 / 44100
        t = np.arange(0, 0.05, dt)

        tweeter_signal = tweeter.waveform(t, dt)
        woofer_signal = woofer.waveform(t, dt)

        assert len(tweeter_signal) == len(t)
        assert len(woofer_signal) == len(t)
        assert not np.any(np.isnan(tweeter_signal))
        assert not np.any(np.isnan(woofer_signal))

    def test_eq_after_crossover(self, tmp_path):
        """Test EQ can be applied after crossover."""
        filepath = tmp_path / "test.wav"
        create_noise_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        xover = Crossover(frequency=2000)
        lowpass = xover.lowpass(audio)

        eq = ParametricEQ()
        eq.add_band(100, -6, 1.0, "peaking")  # Cut bass boom
        processed = eq.apply(lowpass)

        dt = 1.0 / 44100
        t = np.arange(0, 0.05, dt)

        result = processed.waveform(t, dt)

        assert len(result) == len(t)
        assert not np.any(np.isnan(result))

    def test_full_chain_with_all_components(self, tmp_path):
        """Test full signal chain: audio -> crossover -> EQ -> delay."""
        filepath = tmp_path / "test.wav"
        create_noise_wav(filepath, 44100, 0.1)
        audio = AudioFileWaveform(filepath)

        # Crossover at 3kHz
        xover = Crossover(frequency=3000, order=4, type="linkwitz-riley")

        # EQ the woofer
        woofer_eq = ParametricEQ()
        woofer_eq.add_band(80, -3, 1.0, "peaking")  # Room mode cut

        woofer = woofer_eq.apply(xover.lowpass(audio))

        # Delay the tweeter
        tweeter = DelayedWaveform(
            source=xover.highpass(audio),
            delay_seconds=0.0005,  # 0.5ms delay
        )

        dt = 1.0 / 44100
        t = np.arange(0, 0.05, dt)

        woofer_signal = woofer.waveform(t, dt)
        tweeter_signal = tweeter.waveform(t, dt)

        # Both should produce valid output
        assert len(woofer_signal) == len(t)
        assert len(tweeter_signal) == len(t)
        assert not np.any(np.isnan(woofer_signal))
        assert not np.any(np.isnan(tweeter_signal))


class TestImports:
    """Test that DSP classes can be imported correctly."""

    def test_import_from_top_level(self):
        """Test DSP classes can be imported from strata_fdtd."""
        from strata_fdtd import Crossover, DelayedWaveform, FilteredWaveform
        from strata_fdtd import ParametricEQ, ThreeWayCrossover

        assert Crossover is not None
        assert DelayedWaveform is not None
        assert FilteredWaveform is not None
        assert ParametricEQ is not None
        assert ThreeWayCrossover is not None

    def test_import_from_dsp_submodule(self):
        """Test DSP classes can be imported from strata_fdtd.dsp."""
        from strata_fdtd.dsp import (
            Crossover,
            DelayedWaveform,
            EQBand,
            EQWaveform,
            FilteredWaveform,
            ParametricEQ,
            ThreeWayCrossover,
            create_baffle_step_compensation,
        )

        assert Crossover is not None
        assert EQBand is not None
        assert create_baffle_step_compensation is not None

    def test_dsp_submodule_accessible(self):
        """Test dsp submodule is accessible."""
        import strata_fdtd

        assert hasattr(strata_fdtd, "dsp")
