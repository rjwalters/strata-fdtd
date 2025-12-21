"""
Unit tests for audio waveform sources.

Tests verify:
- AudioFileWaveform loading (WAV files)
- Sample rate conversion and resampling
- Stereo to mono conversion
- Start time and duration trimming
- Loop functionality
- Compatibility with GaussianPulse interface
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from strata_fdtd import AudioFileWaveform


class TestAudioFileWaveformLoading:
    """Tests for AudioFileWaveform file loading."""

    def test_load_16bit_wav(self, tmp_path):
        """Test loading 16-bit WAV file."""
        # Create test WAV file
        sr = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        filepath = tmp_path / "test_16bit.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath)

        assert waveform.native_sample_rate == sr
        assert waveform.num_samples == len(data)
        assert abs(waveform.duration_seconds - duration) < 0.001

    def test_load_32bit_float_wav(self, tmp_path):
        """Test loading 32-bit float WAV file."""
        sr = 48000
        duration = 0.1
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        data = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        filepath = tmp_path / "test_32bit.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath)

        assert waveform.native_sample_rate == sr
        assert waveform.num_samples == len(data)

    def test_load_stereo_mix(self, tmp_path):
        """Test loading stereo file and mixing to mono."""
        sr = 44100
        duration = 0.1
        n_samples = int(sr * duration)
        # Left channel: 440Hz, Right channel: 880Hz
        left = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)
        right = (np.sin(2 * np.pi * 880 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)
        data = np.column_stack([left, right])

        filepath = tmp_path / "test_stereo.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath, channel="mix")

        assert waveform.native_sample_rate == sr
        assert waveform.num_samples == n_samples

    def test_load_stereo_left_channel(self, tmp_path):
        """Test loading stereo file and selecting left channel."""
        sr = 44100
        duration = 0.1
        n_samples = int(sr * duration)
        left = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)
        right = (np.sin(2 * np.pi * 880 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)
        data = np.column_stack([left, right])

        filepath = tmp_path / "test_stereo.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath, channel=0)

        assert waveform.num_samples == n_samples

    def test_load_stereo_right_channel(self, tmp_path):
        """Test loading stereo file and selecting right channel."""
        sr = 44100
        duration = 0.1
        n_samples = int(sr * duration)
        left = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)
        right = (np.sin(2 * np.pi * 880 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)
        data = np.column_stack([left, right])

        filepath = tmp_path / "test_stereo.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath, channel=1)

        assert waveform.num_samples == n_samples

    def test_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            AudioFileWaveform("/nonexistent/path/audio.wav")

    def test_invalid_channel(self, tmp_path):
        """Test error handling for invalid channel index."""
        sr = 44100
        duration = 0.1
        n_samples = int(sr * duration)
        left = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)
        right = (np.sin(2 * np.pi * 880 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)
        data = np.column_stack([left, right])

        filepath = tmp_path / "test_stereo.wav"
        wavfile.write(filepath, sr, data)

        with pytest.raises(ValueError, match="Channel 5 requested but file only has 2 channels"):
            AudioFileWaveform(filepath, channel=5)


class TestAudioFileWaveformTrimming:
    """Tests for start time and duration trimming."""

    def test_start_time(self, tmp_path):
        """Test loading with start time offset."""
        sr = 44100
        duration = 1.0
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath, start_time=0.5)

        # Should have half the samples
        expected_samples = int(sr * 0.5)
        assert abs(waveform.num_samples - expected_samples) < 2
        assert abs(waveform.duration_seconds - 0.5) < 0.001

    def test_duration(self, tmp_path):
        """Test loading with duration limit."""
        sr = 44100
        duration = 1.0
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath, duration=0.25)

        expected_samples = int(sr * 0.25)
        assert abs(waveform.num_samples - expected_samples) < 2
        assert abs(waveform.duration_seconds - 0.25) < 0.001

    def test_start_time_and_duration(self, tmp_path):
        """Test loading with both start time and duration."""
        sr = 44100
        duration = 1.0
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath, start_time=0.25, duration=0.25)

        expected_samples = int(sr * 0.25)
        assert abs(waveform.num_samples - expected_samples) < 2

    def test_start_time_past_end(self, tmp_path):
        """Test error when start_time is past end of file."""
        sr = 44100
        duration = 0.1
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        with pytest.raises(ValueError, match="start_time.*is beyond end of file"):
            AudioFileWaveform(filepath, start_time=1.0)


class TestAudioFileWaveformResampling:
    """Tests for resampling to simulation timestep."""

    def test_resample_to_higher_rate(self, tmp_path):
        """Test upsampling to higher simulation rate."""
        sr = 44100
        duration = 0.01
        freq = 1000
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        data = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath)

        # Simulate typical FDTD timestep (dt ~ 8µs = 125 kHz)
        dt = 8e-6
        sim_t = np.arange(0, duration, dt)

        result = waveform.waveform(sim_t, dt)

        assert len(result) == len(sim_t)
        # Check amplitude is reasonable (normalized)
        assert np.max(np.abs(result)) <= 1.1

    def test_resample_preserves_frequency(self, tmp_path):
        """Test that resampling preserves frequency content."""
        sr = 44100
        duration = 0.1
        freq = 440  # A4
        n_samples = int(sr * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        data = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath)

        # Resample to 100 kHz
        dt = 1e-5
        sim_t = np.arange(0, duration, dt)
        result = waveform.waveform(sim_t, dt)

        # Check via FFT that fundamental frequency is preserved
        fft = np.fft.fft(result)
        freqs = np.fft.fftfreq(len(result), dt)
        peak_idx = np.argmax(np.abs(fft[:len(fft)//2]))
        peak_freq = abs(freqs[peak_idx])

        # Allow 5% tolerance
        assert abs(peak_freq - freq) < freq * 0.05

    def test_resample_caching(self, tmp_path):
        """Test that resampled results are cached."""
        sr = 44100
        duration = 0.01
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath)

        dt = 1e-5
        sim_t = np.arange(0, duration, dt)

        # First call
        result1 = waveform.waveform(sim_t, dt)
        # Second call should use cache
        result2 = waveform.waveform(sim_t, dt)

        np.testing.assert_array_equal(result1, result2)


class TestAudioFileWaveformInterface:
    """Tests for GaussianPulse-compatible interface."""

    def test_waveform_method_signature(self, tmp_path):
        """Test that waveform method has correct signature."""
        sr = 44100
        duration = 0.1
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath)

        # Test waveform method works like GaussianPulse
        dt = 1e-5
        t = np.array([0.0, 0.001, 0.002])

        result = waveform.waveform(t, dt)

        assert isinstance(result, np.ndarray)
        assert result.shape == t.shape

    def test_amplitude_scaling(self, tmp_path):
        """Test amplitude parameter scales output."""
        sr = 44100
        duration = 0.01
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform1 = AudioFileWaveform(filepath, amplitude=1.0)
        waveform2 = AudioFileWaveform(filepath, amplitude=0.5)

        dt = 1e-5
        t = np.arange(0, duration, dt)

        result1 = waveform1.waveform(t, dt)
        result2 = waveform2.waveform(t, dt)

        # Amplitude 0.5 should give half the values
        np.testing.assert_allclose(result2, result1 * 0.5, rtol=1e-5)

    def test_past_end_returns_zero(self, tmp_path):
        """Test that requesting times past end returns zero."""
        sr = 44100
        duration = 0.1
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath, loop=False)

        dt = 1e-5
        # Request times well past end of file
        t = np.array([0.5, 1.0, 2.0])

        result = waveform.waveform(t, dt)

        np.testing.assert_array_equal(result, np.zeros_like(t))


class TestAudioFileWaveformLoop:
    """Tests for loop functionality."""

    def test_loop_wraps_around(self, tmp_path):
        """Test that loop=True wraps audio around."""
        sr = 44100
        duration = 0.1
        n_samples = int(sr * duration)
        # Create distinctive pattern
        data = np.linspace(-1, 1, n_samples).astype(np.float32)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath, loop=True)

        dt = 1.0 / sr  # Use native sample rate for simplicity

        # Get value at t=0
        t0 = np.array([0.0])
        val0 = waveform.waveform(t0, dt)[0]

        # Get value at t=duration (should wrap to start)
        t1 = np.array([duration])
        val1 = waveform.waveform(t1, dt)[0]

        # Should be approximately equal (same position after wrap)
        assert abs(val0 - val1) < 0.01

    def test_no_loop_stops_at_end(self, tmp_path):
        """Test that loop=False returns zero past end."""
        sr = 44100
        duration = 0.1
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath, loop=False)

        dt = 1e-5
        t = np.array([duration + 0.01])  # Past end

        result = waveform.waveform(t, dt)

        assert result[0] == 0.0


class TestAudioFileWaveformProperties:
    """Tests for property accessors."""

    def test_duration_seconds(self, tmp_path):
        """Test duration_seconds property."""
        sr = 44100
        duration = 0.5
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath)

        assert abs(waveform.duration_seconds - duration) < 0.001

    def test_native_sample_rate(self, tmp_path):
        """Test native_sample_rate property."""
        sr = 48000
        duration = 0.1
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath)

        assert waveform.native_sample_rate == sr

    def test_num_samples(self, tmp_path):
        """Test num_samples property."""
        sr = 44100
        duration = 0.1
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath)

        assert waveform.num_samples == n_samples


class TestAudioFileWaveformIntegration:
    """Integration tests with other strata_fdtd components."""

    def test_with_circular_membrane_source(self, tmp_path):
        """Test AudioFileWaveform works with CircularMembraneSource."""
        from strata_fdtd import CircularMembraneSource

        sr = 44100
        duration = 0.01
        n_samples = int(sr * duration)
        data = (np.sin(2 * np.pi * 440 * np.linspace(0, duration, n_samples)) * 32767).astype(np.int16)

        filepath = tmp_path / "test.wav"
        wavfile.write(filepath, sr, data)

        waveform = AudioFileWaveform(filepath)

        # Should be usable as waveform for CircularMembraneSource
        source = CircularMembraneSource(
            center=(0.1, 0.01, 0.15),
            radius=0.1,
            normal_axis='y',
            waveform=waveform,
        )

        assert source.waveform is waveform

    def test_import_from_top_level(self):
        """Test AudioFileWaveform can be imported from strata_fdtd."""
        from strata_fdtd import AudioFileWaveform as AWF

        assert AWF is AudioFileWaveform
