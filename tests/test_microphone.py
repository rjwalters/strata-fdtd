"""
Unit tests for the Microphone class in FDTD simulation.

Tests verify:
- Microphone initialization and position validation
- Trilinear interpolation accuracy
- Recording during simulation
- WAV export functionality
- Integration with FDTDSolver
"""

import os
import tempfile
import wave

import numpy as np
import pytest

from strata_fdtd import FDTDSolver, GaussianPulse, Microphone

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_solver():
    """Create a small solver for fast tests."""
    return FDTDSolver(shape=(20, 20, 20), resolution=5e-3, c=343.0, rho=1.2)


@pytest.fixture
def medium_solver():
    """Create a medium solver for validation tests."""
    return FDTDSolver(shape=(50, 50, 50), resolution=2e-3, c=343.0, rho=1.2)


# =============================================================================
# Microphone Initialization Tests
# =============================================================================


class TestMicrophoneInitialization:
    def test_microphone_creation(self):
        """Test basic microphone creation."""
        mic = Microphone(position=(0.05, 0.05, 0.05), name="test")
        assert mic.position == (0.05, 0.05, 0.05)
        assert mic.name == "test"
        assert len(mic) == 0

    def test_microphone_without_name(self):
        """Test microphone creation without name."""
        mic = Microphone(position=(0.01, 0.02, 0.03))
        assert mic.position == (0.01, 0.02, 0.03)
        assert mic.name is None

    def test_add_microphone_with_position(self, small_solver):
        """Test adding microphone by position tuple."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="center"
        )
        assert mic.name == "center"
        assert "center" in small_solver.microphones

    def test_add_microphone_with_object(self, small_solver):
        """Test adding microphone by Microphone object."""
        mic = Microphone(position=(0.05, 0.05, 0.05), name="center")
        returned_mic = small_solver.add_microphone(mic)
        assert returned_mic is mic
        assert "center" in small_solver.microphones

    def test_add_microphone_auto_name(self, small_solver):
        """Test automatic name generation."""
        mic = small_solver.add_microphone(position=(0.05, 0.05, 0.05))
        assert mic.name == "mic_0"

        mic2 = small_solver.add_microphone(position=(0.06, 0.05, 0.05))
        assert mic2.name == "mic_1"

    def test_duplicate_name_rejected(self, small_solver):
        """Test that duplicate microphone names are rejected."""
        small_solver.add_microphone(position=(0.05, 0.05, 0.05), name="test")
        with pytest.raises(ValueError, match="already exists"):
            small_solver.add_microphone(position=(0.06, 0.05, 0.05), name="test")

    def test_position_outside_domain_rejected(self, small_solver):
        """Test that positions outside domain are rejected."""
        # Domain is 20*5mm = 100mm in each dimension
        with pytest.raises(ValueError, match="outside simulation domain"):
            small_solver.add_microphone(position=(0.15, 0.05, 0.05), name="far")

    def test_position_at_boundary_rejected(self, small_solver):
        """Test that positions at domain boundary are rejected."""
        # Position at edge (would need interpolation outside domain)
        max_pos = (small_solver.shape[0] - 1) * small_solver.dx
        with pytest.raises(ValueError, match="outside simulation domain"):
            small_solver.add_microphone(position=(max_pos, 0.05, 0.05), name="edge")


# =============================================================================
# Trilinear Interpolation Tests
# =============================================================================


class TestTrilinearInterpolation:
    def test_interpolation_at_grid_point(self, small_solver):
        """Test that interpolation at grid point returns exact value."""
        # Position exactly at grid point (5, 5, 5)
        pos = (5 * small_solver.dx, 5 * small_solver.dx, 5 * small_solver.dx)
        mic = small_solver.add_microphone(position=pos, name="grid_aligned")

        # Set known pressure at that grid point
        small_solver.p[5, 5, 5] = 1.0

        # Manually trigger recording to test interpolation directly
        # (before step() modifies pressure field)
        mic.record(small_solver.p, 0.0)

        waveform = mic.get_waveform()
        # At exact grid point, interpolation should return exact value
        assert waveform[0] == pytest.approx(1.0, rel=0.01)

    def test_interpolation_at_cell_center(self, small_solver):
        """Test interpolation at cell center (equal weights)."""
        # Position at center of cell (5.5, 5.5, 5.5) in grid coords
        pos = (5.5 * small_solver.dx, 5.5 * small_solver.dx, 5.5 * small_solver.dx)
        mic = small_solver.add_microphone(position=pos, name="cell_center")

        # Set uniform pressure in surrounding cells
        small_solver.p[5:7, 5:7, 5:7] = 1.0

        # Manually trigger recording to test interpolation directly
        mic.record(small_solver.p, 0.0)

        waveform = mic.get_waveform()
        # Average of 8 equal values should be 1.0
        assert waveform[0] == pytest.approx(1.0, rel=0.01)

    def test_interpolation_weighted(self, small_solver):
        """Test that interpolation weights are applied correctly."""
        # Position at (5.25, 5, 5) in grid coords (25% along x)
        pos = (5.25 * small_solver.dx, 5 * small_solver.dx, 5 * small_solver.dx)
        mic = small_solver.add_microphone(position=pos, name="weighted")

        # Set different pressures: p[5]=0, p[6]=4
        small_solver.p[5, 5, 5] = 0.0
        small_solver.p[6, 5, 5] = 4.0

        # Manually trigger recording to test interpolation directly
        mic.record(small_solver.p, 0.0)

        waveform = mic.get_waveform()
        # Expected: 0.75 * 0 + 0.25 * 4 = 1.0
        assert waveform[0] == pytest.approx(1.0, rel=0.05)


# =============================================================================
# Recording Tests
# =============================================================================


class TestRecording:
    def test_recording_during_simulation(self, small_solver):
        """Test that microphone records during simulation steps."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )

        for _ in range(10):
            small_solver.step()

        assert len(mic) == 10

    def test_time_axis_correct(self, small_solver):
        """Test that time axis is correct."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )

        n_steps = 10
        for _ in range(n_steps):
            small_solver.step()

        times = mic.get_time_axis()
        assert len(times) == n_steps

        # Check time values
        expected_times = np.arange(n_steps) * small_solver.dt
        np.testing.assert_allclose(times, expected_times, rtol=1e-10)

    def test_sample_rate(self, small_solver):
        """Test sample rate calculation."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )

        expected_sr = 1.0 / small_solver.dt
        assert mic.get_sample_rate() == pytest.approx(expected_sr)

    def test_multiple_microphones(self, small_solver):
        """Test recording with multiple microphones."""
        mic1 = small_solver.add_microphone(
            position=(0.03, 0.05, 0.05), name="mic1"
        )
        mic2 = small_solver.add_microphone(
            position=(0.07, 0.05, 0.05), name="mic2"
        )

        for _ in range(10):
            small_solver.step()

        assert len(mic1) == 10
        assert len(mic2) == 10

    def test_clear_recording(self, small_solver):
        """Test clearing recorded data."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )

        for _ in range(10):
            small_solver.step()

        assert len(mic) == 10
        mic.clear()
        assert len(mic) == 0

    def test_reset_clears_microphones(self, small_solver):
        """Test that solver reset clears microphone data."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )

        for _ in range(10):
            small_solver.step()

        assert len(mic) == 10
        small_solver.reset()
        assert len(mic) == 0


# =============================================================================
# Waveform Access Tests
# =============================================================================


class TestWaveformAccess:
    def test_get_waveform_dtype(self, small_solver):
        """Test waveform array dtype."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )
        small_solver.step()

        waveform = mic.get_waveform()
        assert waveform.dtype == np.float32

    def test_get_time_axis_dtype(self, small_solver):
        """Test time axis array dtype."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )
        small_solver.step()

        times = mic.get_time_axis()
        assert times.dtype == np.float64

    def test_waveform_records_actual_pressure(self, medium_solver):
        """Test that waveform captures actual acoustic signal."""
        # Add source
        medium_solver.add_source(GaussianPulse(
            position=(10, 25, 25),
            frequency=1000,
            amplitude=10.0,  # Increase amplitude for better detection
        ))

        # Add microphone near source (closer to source)
        mic = medium_solver.add_microphone(
            position=(0.025, 0.05, 0.05), name="near_source"
        )

        # Run simulation longer to capture the pulse
        medium_solver.run(duration=0.005)

        waveform = mic.get_waveform()

        # Should have recorded something non-zero
        # The Gaussian pulse has an envelope, so max amplitude may be lower
        assert np.max(np.abs(waveform)) > 1e-5


# =============================================================================
# WAV Export Tests
# =============================================================================


class TestWAVExport:
    def test_wav_export_basic(self, small_solver):
        """Test basic WAV export."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )

        # Generate some data
        small_solver.p[10, 10, 10] = 1.0
        for _ in range(100):
            small_solver.step()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            filepath = f.name

        try:
            mic.to_wav(filepath, sample_rate=44100)

            # Verify file exists and is valid WAV
            with wave.open(filepath, 'rb') as wav_file:
                assert wav_file.getnchannels() == 1
                assert wav_file.getframerate() == 44100
                assert wav_file.getsampwidth() == 2  # 16-bit default
        finally:
            os.unlink(filepath)

    def test_wav_export_32bit(self, small_solver):
        """Test 32-bit WAV export."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )

        small_solver.p[10, 10, 10] = 1.0
        for _ in range(100):
            small_solver.step()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            filepath = f.name

        try:
            mic.to_wav(filepath, sample_rate=48000, bit_depth=32)

            with wave.open(filepath, 'rb') as wav_file:
                assert wav_file.getsampwidth() == 4  # 32-bit
                assert wav_file.getframerate() == 48000
        finally:
            os.unlink(filepath)

    def test_wav_export_no_normalize(self, small_solver):
        """Test WAV export without normalization."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )

        small_solver.p[10, 10, 10] = 1.0
        for _ in range(100):
            small_solver.step()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            filepath = f.name

        try:
            mic.to_wav(filepath, normalize=False)
            # Should not raise
            assert os.path.exists(filepath)
        finally:
            os.unlink(filepath)

    def test_wav_export_empty_raises(self, small_solver):
        """Test that WAV export raises with no data."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )

        with pytest.raises(RuntimeError, match="No data recorded"):
            mic.to_wav("test.wav")

    def test_wav_export_invalid_bit_depth(self, small_solver):
        """Test that invalid bit depth raises."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )
        small_solver.step()

        with pytest.raises(ValueError, match="bit_depth must be"):
            mic.to_wav("test.wav", bit_depth=24)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    def test_microphone_vs_probe_consistency(self, medium_solver):
        """Test that microphone at grid point matches probe."""
        # Add probe and microphone at same location
        grid_pos = (25, 25, 25)
        phys_pos = tuple(i * medium_solver.dx for i in grid_pos)

        medium_solver.add_probe('probe', position=grid_pos)
        mic = medium_solver.add_microphone(position=phys_pos, name="mic")

        # Add source
        medium_solver.add_source(GaussianPulse(
            position=(10, 25, 25),
            frequency=2000,
        ))

        # Run simulation
        medium_solver.run(duration=0.001)

        probe_data = medium_solver.get_probe_data('probe')['probe']
        mic_data = mic.get_waveform()

        # Should be very close (not exact due to float precision)
        np.testing.assert_allclose(probe_data, mic_data, rtol=1e-5)

    def test_wave_arrival_time(self, medium_solver):
        """Test that wave arrival time is physically correct."""
        # Source at one end with higher amplitude
        medium_solver.add_source(GaussianPulse(
            position=(5, 25, 25),
            frequency=2000,
            amplitude=100.0,  # Higher amplitude for clearer detection
        ))

        # Microphones at different distances (both well within domain)
        mic_near = medium_solver.add_microphone(
            position=(0.02, 0.05, 0.05), name="near"  # 20mm from origin
        )
        mic_far = medium_solver.add_microphone(
            position=(0.06, 0.05, 0.05), name="far"   # 60mm from origin
        )

        # Run longer to capture wave propagation
        medium_solver.run(duration=0.001)

        near_data = mic_near.get_waveform()
        far_data = mic_far.get_waveform()

        # Find first significant signal using a relative threshold
        near_max = np.max(np.abs(near_data))
        far_max = np.max(np.abs(far_data))

        # Use 1% of peak as threshold
        near_threshold = near_max * 0.01
        far_threshold = far_max * 0.01

        near_arrival = np.argmax(np.abs(near_data) > near_threshold)
        far_arrival = np.argmax(np.abs(far_data) > far_threshold)

        # Far microphone should see signal later
        assert far_arrival > near_arrival, (
            f"Expected far arrival ({far_arrival}) > near arrival ({near_arrival})"
        )

    def test_microphones_property_access(self, small_solver):
        """Test dict-like access to microphones."""
        small_solver.add_microphone(position=(0.03, 0.05, 0.05), name="mic1")
        small_solver.add_microphone(position=(0.05, 0.05, 0.05), name="mic2")

        assert len(small_solver.microphones) == 2
        assert "mic1" in small_solver.microphones
        assert "mic2" in small_solver.microphones

        for name, mic in small_solver.microphones.items():
            assert isinstance(mic, Microphone)
            assert mic.name == name

    def test_repr(self, small_solver):
        """Test microphone string representation."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05), name="test"
        )

        repr_str = repr(mic)
        assert "test" in repr_str
        assert "0.05" in repr_str
        assert "samples=0" in repr_str

        small_solver.step()
        repr_str = repr(mic)
        assert "samples=1" in repr_str


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    def test_uninitialized_sample_rate(self):
        """Test that sample rate raises before initialization."""
        mic = Microphone(position=(0.05, 0.05, 0.05))
        with pytest.raises(RuntimeError, match="not initialized"):
            mic.get_sample_rate()

    def test_uninitialized_record(self):
        """Test that record raises before initialization."""
        mic = Microphone(position=(0.05, 0.05, 0.05))
        with pytest.raises(RuntimeError, match="not initialized"):
            mic.record(np.zeros((10, 10, 10)), 0.0)
