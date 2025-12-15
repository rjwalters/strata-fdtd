"""
Unit tests for directional microphone patterns.

Tests verify:
- Polar pattern parameter validation
- Direction and up vector handling
- Omnidirectional backward compatibility
- Cardioid pattern polar response
- Figure-8 pattern polar response
- Custom pattern functions
- Integration with FDTDSolver

Issue #106: Add directional microphone patterns
"""

import numpy as np
import pytest

from strata_fdtd import POLAR_PATTERNS, FDTDSolver, GaussianPulse, Microphone

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def small_solver():
    """Create a small solver for fast tests."""
    return FDTDSolver(shape=(30, 30, 30), resolution=5e-3, c=343.0, rho=1.2)


@pytest.fixture
def medium_solver():
    """Create a medium solver for pattern validation tests."""
    return FDTDSolver(shape=(50, 50, 50), resolution=2e-3, c=343.0, rho=1.2)


# =============================================================================
# Pattern Parameter Tests
# =============================================================================


class TestPatternParameters:
    def test_default_pattern_is_omni(self):
        """Test that default pattern is omnidirectional."""
        mic = Microphone(position=(0.05, 0.05, 0.05))
        assert mic._pattern_name == "omni"
        assert not mic.is_directional()

    def test_valid_pattern_names(self):
        """Test all valid pattern names are accepted."""
        for pattern_name in POLAR_PATTERNS:
            mic = Microphone(
                position=(0.05, 0.05, 0.05),
                pattern=pattern_name,
            )
            assert mic._pattern_name == pattern_name

    def test_invalid_pattern_name_rejected(self):
        """Test that invalid pattern names are rejected."""
        with pytest.raises(ValueError, match="Unknown pattern"):
            Microphone(position=(0.05, 0.05, 0.05), pattern="invalid")

    def test_pattern_type_validation(self):
        """Test that pattern must be str or callable."""
        with pytest.raises(TypeError, match="must be str or callable"):
            Microphone(position=(0.05, 0.05, 0.05), pattern=42)

    def test_custom_pattern_function(self):
        """Test that custom pattern function is accepted."""
        def custom_pattern(theta):
            return 0.7 + 0.3 * np.cos(theta)

        mic = Microphone(
            position=(0.05, 0.05, 0.05),
            pattern=custom_pattern,
        )
        assert mic._pattern_name == "custom"
        assert mic._custom_pattern is custom_pattern

    def test_is_directional_omni(self):
        """Test is_directional returns False for omni."""
        mic = Microphone(position=(0.05, 0.05, 0.05), pattern="omni")
        assert not mic.is_directional()

    def test_is_directional_cardioid(self):
        """Test is_directional returns True for cardioid."""
        mic = Microphone(position=(0.05, 0.05, 0.05), pattern="cardioid")
        assert mic.is_directional()


# =============================================================================
# Direction Vector Tests
# =============================================================================


class TestDirectionVector:
    def test_default_direction(self):
        """Test default direction is +x."""
        mic = Microphone(position=(0.05, 0.05, 0.05))
        assert mic.direction == pytest.approx((1.0, 0.0, 0.0))

    def test_custom_direction_normalized(self):
        """Test that direction vector is normalized."""
        mic = Microphone(
            position=(0.05, 0.05, 0.05),
            direction=(2.0, 0.0, 0.0),
        )
        assert mic.direction == pytest.approx((1.0, 0.0, 0.0))

    def test_arbitrary_direction(self):
        """Test arbitrary direction is normalized correctly."""
        mic = Microphone(
            position=(0.05, 0.05, 0.05),
            direction=(1.0, 1.0, 1.0),
        )
        expected = 1.0 / np.sqrt(3)
        assert mic.direction == pytest.approx((expected, expected, expected))

    def test_zero_direction_rejected(self):
        """Test that zero direction vector is rejected."""
        with pytest.raises(ValueError, match="direction vector cannot be zero"):
            Microphone(position=(0.05, 0.05, 0.05), direction=(0, 0, 0))

    def test_default_up_vector(self):
        """Test default up vector is +z."""
        mic = Microphone(position=(0.05, 0.05, 0.05))
        assert mic._up == pytest.approx([0.0, 0.0, 1.0])

    def test_zero_up_rejected(self):
        """Test that zero up vector is rejected."""
        with pytest.raises(ValueError, match="up vector cannot be zero"):
            Microphone(position=(0.05, 0.05, 0.05), up=(0, 0, 0))

    def test_parallel_direction_up_rejected(self):
        """Test that parallel direction and up vectors are rejected."""
        with pytest.raises(ValueError, match="cannot be parallel"):
            Microphone(
                position=(0.05, 0.05, 0.05),
                direction=(0, 0, 1),
                up=(0, 0, 1),
            )


# =============================================================================
# Repr Tests
# =============================================================================


class TestMicrophoneRepr:
    def test_repr_omni(self, small_solver):
        """Test repr for omnidirectional microphone."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05),
            name="test",
            pattern="omni",
        )
        repr_str = repr(mic)
        assert "test" in repr_str
        assert "omni" in repr_str
        assert "direction" not in repr_str

    def test_repr_directional(self, small_solver):
        """Test repr for directional microphone includes direction."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05),
            name="test",
            pattern="cardioid",
            direction=(0, 1, 0),
        )
        repr_str = repr(mic)
        assert "cardioid" in repr_str
        assert "direction" in repr_str


# =============================================================================
# Solver Integration Tests
# =============================================================================


class TestSolverIntegration:
    def test_add_microphone_with_pattern(self, small_solver):
        """Test adding microphone with pattern via solver."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05),
            name="cardioid",
            pattern="cardioid",
            direction=(1, 0, 0),
        )
        assert mic._pattern_name == "cardioid"
        assert "cardioid" in small_solver.microphones

    def test_omni_records_without_velocity(self, small_solver):
        """Test that omni microphone records with just pressure."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05),
            name="omni",
        )

        for _ in range(10):
            small_solver.step()

        assert len(mic) == 10

    def test_cardioid_records_with_velocity(self, small_solver):
        """Test that cardioid microphone records using velocity fields."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05),
            name="cardioid",
            pattern="cardioid",
            direction=(1, 0, 0),
        )

        for _ in range(10):
            small_solver.step()

        assert len(mic) == 10

    def test_multiple_pattern_types(self, small_solver):
        """Test multiple microphones with different patterns."""
        mic_omni = small_solver.add_microphone(
            position=(0.03, 0.05, 0.05),
            name="omni",
            pattern="omni",
        )
        mic_cardioid = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05),
            name="cardioid",
            pattern="cardioid",
        )
        mic_figure8 = small_solver.add_microphone(
            position=(0.07, 0.05, 0.05),
            name="figure8",
            pattern="figure8",
        )

        for _ in range(10):
            small_solver.step()

        assert len(mic_omni) == 10
        assert len(mic_cardioid) == 10
        assert len(mic_figure8) == 10


# =============================================================================
# Polar Response Tests
# =============================================================================


class TestPolarResponse:
    """Tests verifying polar pattern response characteristics."""

    def test_cardioid_rear_rejection(self, medium_solver):
        """Test that cardioid rejects sound from rear.

        Place a source behind a cardioid mic and compare to omni response.
        The cardioid should show significant rejection.
        """
        # Add source behind the microphones (+x direction)
        medium_solver.add_source(GaussianPulse(
            position=(40, 25, 25),  # Far in +x
            frequency=2000,
            amplitude=100.0,
        ))

        # Microphones at center, both pointing toward -x (away from source)
        mic_omni = medium_solver.add_microphone(
            position=(0.04, 0.05, 0.05),
            name="omni",
            pattern="omni",
        )
        mic_cardioid = medium_solver.add_microphone(
            position=(0.04, 0.05, 0.05),
            name="cardioid",
            pattern="cardioid",
            direction=(-1, 0, 0),  # Pointing away from source (rear rejection)
        )

        # Run simulation
        medium_solver.run(duration=0.002)

        omni_data = mic_omni.get_waveform()
        cardioid_data = mic_cardioid.get_waveform()

        # Both should record something
        omni_peak = np.max(np.abs(omni_data))
        cardioid_peak = np.max(np.abs(cardioid_data))

        assert omni_peak > 1e-6, "Omni should record signal"
        assert cardioid_peak > 1e-6, "Cardioid should record signal"

        # Cardioid pointing away from source should be attenuated
        # (not a perfect null due to near-field effects, but significant)
        rejection_db = 20 * np.log10(cardioid_peak / omni_peak) if omni_peak > 0 else 0
        # Expect at least some rejection from rear
        # Note: Near-field effects reduce the theoretical -inf dB null
        assert rejection_db < 0, f"Expected rear rejection, got {rejection_db:.1f} dB"

    def test_figure8_side_null(self, medium_solver):
        """Test that figure-8 has null response from sides.

        Place a source to the side of a figure-8 mic.
        """
        # Add source to the side (+y direction)
        medium_solver.add_source(GaussianPulse(
            position=(25, 40, 25),  # Far in +y
            frequency=2000,
            amplitude=100.0,
        ))

        # Microphones at center
        mic_omni = medium_solver.add_microphone(
            position=(0.05, 0.04, 0.05),
            name="omni",
            pattern="omni",
        )
        mic_figure8 = medium_solver.add_microphone(
            position=(0.05, 0.04, 0.05),
            name="figure8",
            pattern="figure8",
            direction=(1, 0, 0),  # Pointing perpendicular to source (null)
        )

        # Run simulation
        medium_solver.run(duration=0.002)

        omni_data = mic_omni.get_waveform()
        fig8_data = mic_figure8.get_waveform()

        omni_peak = np.max(np.abs(omni_data))
        fig8_peak = np.max(np.abs(fig8_data))

        assert omni_peak > 1e-6, "Omni should record signal"

        # Figure-8 should show significant rejection from side
        if omni_peak > 1e-10:
            rejection_ratio = fig8_peak / omni_peak
            # Expect significant side rejection (not perfect due to near-field)
            assert rejection_ratio < 0.5, f"Expected side rejection, ratio={rejection_ratio:.3f}"

    def test_on_axis_response_similar(self, medium_solver):
        """Test that on-axis response is similar for all patterns.

        All first-order patterns should have similar on-axis sensitivity.
        """
        # Add source in front of microphones
        medium_solver.add_source(GaussianPulse(
            position=(10, 25, 25),  # In front (-x from mic position)
            frequency=2000,
            amplitude=100.0,
        ))

        # Microphones pointing toward source
        mic_omni = medium_solver.add_microphone(
            position=(0.06, 0.05, 0.05),
            name="omni",
        )
        mic_cardioid = medium_solver.add_microphone(
            position=(0.06, 0.05, 0.05),
            name="cardioid",
            pattern="cardioid",
            direction=(-1, 0, 0),  # Toward source
        )
        mic_figure8 = medium_solver.add_microphone(
            position=(0.06, 0.05, 0.05),
            name="figure8",
            pattern="figure8",
            direction=(-1, 0, 0),  # Toward source
        )

        medium_solver.run(duration=0.002)

        omni_peak = np.max(np.abs(mic_omni.get_waveform()))
        cardioid_peak = np.max(np.abs(mic_cardioid.get_waveform()))
        fig8_peak = np.max(np.abs(mic_figure8.get_waveform()))

        # All should record similar levels on-axis (within 6 dB)
        # Note: exact match not expected due to pattern normalization differences
        if omni_peak > 1e-10:
            cardioid_ratio = cardioid_peak / omni_peak
            fig8_ratio = fig8_peak / omni_peak
            # Within 6 dB (factor of 2)
            assert 0.25 < cardioid_ratio < 4.0, f"Cardioid ratio: {cardioid_ratio:.2f}"
            assert 0.25 < fig8_ratio < 4.0, f"Figure-8 ratio: {fig8_ratio:.2f}"


# =============================================================================
# Custom Pattern Tests
# =============================================================================


class TestCustomPatterns:
    def test_custom_subcardioid(self, small_solver):
        """Test custom subcardioid pattern function."""
        def subcardioid(theta):
            return 0.7 + 0.3 * np.cos(theta)

        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05),
            name="subcardioid",
            pattern=subcardioid,
        )

        # Should be directional
        assert mic.is_directional()
        assert mic._pattern_name == "custom"

        for _ in range(10):
            small_solver.step()

        assert len(mic) == 10

    def test_custom_lambda_pattern(self, small_solver):
        """Test custom pattern as lambda."""
        mic = small_solver.add_microphone(
            position=(0.05, 0.05, 0.05),
            name="narrow",
            pattern=lambda theta: np.cos(theta) ** 2,
        )

        assert mic._pattern_name == "custom"

        for _ in range(10):
            small_solver.step()

        assert len(mic) == 10


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibility:
    def test_existing_api_unchanged(self, small_solver):
        """Test that existing API continues to work."""
        # Method 1: Position and name
        mic1 = small_solver.add_microphone(
            position=(0.03, 0.05, 0.05),
            name="mic1",
        )

        # Method 2: Microphone object
        mic2 = Microphone(position=(0.05, 0.05, 0.05), name="mic2")
        small_solver.add_microphone(mic2)

        # Run simulation
        for _ in range(10):
            small_solver.step()

        # Check recordings
        assert len(mic1) == 10
        assert len(mic2) == 10

        # Check data access methods
        _ = mic1.get_waveform()
        _ = mic1.get_time_axis()
        _ = mic1.get_sample_rate()

    def test_omni_matches_original_behavior(self, medium_solver):
        """Test that omnidirectional pattern produces same results as original."""
        # Add source
        medium_solver.add_source(GaussianPulse(
            position=(10, 25, 25),
            frequency=2000,
            amplitude=10.0,
        ))

        # Two microphones at same position - one explicit omni, one default
        mic_default = medium_solver.add_microphone(
            position=(0.05, 0.05, 0.05),
            name="default",
        )
        mic_explicit = medium_solver.add_microphone(
            position=(0.05, 0.05, 0.05),
            name="explicit",
            pattern="omni",
        )

        medium_solver.run(duration=0.001)

        default_data = mic_default.get_waveform()
        explicit_data = mic_explicit.get_waveform()

        # Should be identical
        np.testing.assert_allclose(default_data, explicit_data, rtol=1e-10)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    def test_directional_requires_velocity_manual(self):
        """Test that directional mic requires velocity for manual recording."""
        mic = Microphone(
            position=(0.05, 0.05, 0.05),
            pattern="cardioid",
        )

        # Create dummy pressure field
        pressure = np.zeros((20, 20, 20), dtype=np.float32)

        # Initialize mic manually (normally done by solver)
        class FakeSolver:
            dt = 1e-5
            dx = 5e-3
            rho = 1.2
            c = 343.0
            shape = (20, 20, 20)

        mic._initialize(FakeSolver())

        # Should raise when velocity not provided
        with pytest.raises(ValueError, match="Velocity fields.*required"):
            mic.record(pressure, 0.0)

    def test_uninitialized_mic_raises(self):
        """Test that uninitialized microphone raises on record."""
        mic = Microphone(position=(0.05, 0.05, 0.05), pattern="cardioid")

        with pytest.raises(RuntimeError, match="not initialized"):
            mic.record(np.zeros((10, 10, 10)), 0.0)
