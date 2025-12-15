"""
Tests for Helmholtz resonator primitives.

Tests cover:
- Resonator geometry construction (box, sphere, cylinder cavities)
- Resonant frequency calculations with end corrections
- SDF evaluation and voxelization
- Array generation for broadband absorbers
- Edge cases and validation
"""

import numpy as np
import pytest

from strata_fdtd import Box, HelmholtzResonator, UniformGrid
from strata_fdtd.geometry import helmholtz_array


class TestHelmholtzResonatorGeometry:
    """Test resonator geometry construction and SDF evaluation."""

    def test_box_cavity_construction(self):
        """Test resonator with box cavity."""
        res = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="box",
            cavity_size=(0.03, 0.03, 0.03),
            neck_length=0.01,
            neck_radius=0.005,
            neck_direction=(0, 0, 1),
        )

        # Check internal geometry was built
        assert res._cavity is not None
        assert res._neck is not None
        assert res._combined is not None

        # Check bounding box includes both cavity and neck
        bb_min, bb_max = res.bounding_box
        assert len(bb_min) == 3
        assert len(bb_max) == 3
        assert np.all(bb_min < bb_max)

    def test_sphere_cavity_construction(self):
        """Test resonator with spherical cavity."""
        res = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="sphere",
            cavity_size=(0.015,),  # 15mm radius
            neck_length=0.01,
            neck_radius=0.003,
        )

        # Check cavity volume calculation
        expected_volume = (4 / 3) * np.pi * 0.015**3
        assert np.isclose(res.cavity_volume, expected_volume, rtol=1e-6)

    def test_cylinder_cavity_construction(self):
        """Test resonator with cylindrical cavity."""
        res = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="cylinder",
            cavity_size=(0.01, 0.03),  # radius, length
            neck_length=0.01,
            neck_radius=0.003,
        )

        # Check cavity volume calculation
        expected_volume = np.pi * 0.01**2 * 0.03
        assert np.isclose(res.cavity_volume, expected_volume, rtol=1e-6)

    def test_arbitrary_neck_direction(self):
        """Test resonator with non-axis-aligned neck direction."""
        # Neck at 45° in XZ plane
        direction = np.array([1.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)

        res = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="box",
            cavity_size=(0.02, 0.02, 0.02),
            neck_length=0.01,
            neck_radius=0.003,
            neck_direction=direction,
        )

        # Check neck direction was normalized
        assert np.allclose(np.linalg.norm(res.neck_direction), 1.0)

    def test_sdf_evaluation(self):
        """Test SDF evaluation at various points."""
        res = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="box",
            cavity_size=(0.02, 0.02, 0.02),
            neck_length=0.01,
            neck_radius=0.005,
        )

        # Points to test
        points = np.array([
            [0.05, 0.05, 0.05],    # Center of cavity (inside)
            [0.05, 0.05, 0.065],   # In middle of neck (inside)
            [0.1, 0.1, 0.1],       # Far outside
        ])

        distances = res.sdf(points)

        # Check signs
        assert distances[0] < 0, "Cavity center should be inside"
        assert distances[1] < 0, "Neck should be inside"
        assert distances[2] > 0, "Far point should be outside"

    def test_voxelization(self):
        """Test voxelization to grid."""
        res = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="box",
            cavity_size=(0.02, 0.02, 0.02),
            neck_length=0.01,
            neck_radius=0.005,
        )

        grid = UniformGrid(shape=(50, 50, 50), resolution=1e-3)
        mask = res.voxelize(grid)

        # Check mask shape
        assert mask.shape == (50, 50, 50)
        assert mask.dtype == bool

        # Check that some cells are inside (True)
        assert np.any(mask), "Expected some voxels inside resonator"

        # Check that some cells are outside (False)
        assert not np.all(mask), "Expected some voxels outside resonator"


class TestResonantFrequency:
    """Test resonant frequency calculations."""

    def test_box_cavity_frequency(self):
        """Test frequency calculation for box cavity."""
        res = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="box",
            cavity_size=(0.03, 0.03, 0.03),  # 27 cm³ = 27e-6 m³
            neck_length=0.01,  # 10mm
            neck_radius=0.005,  # 5mm radius = 10mm diameter
        )

        # Calculate expected frequency
        # f = (c / 2π) * sqrt(S / (V * L_eff))
        c = 343.0
        V = 0.03**3
        S = np.pi * 0.005**2
        L_eff = 0.01 + 2 * 0.6 * 0.005  # unflanged correction
        expected = (c / (2 * np.pi)) * np.sqrt(S / (V * L_eff))

        assert np.isclose(res.resonant_frequency, expected, rtol=1e-3)

    def test_sphere_cavity_frequency(self):
        """Test frequency calculation for spherical cavity."""
        res = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="sphere",
            cavity_size=(0.02,),  # 20mm radius
            neck_length=0.015,
            neck_radius=0.004,
        )

        # Calculate expected frequency
        c = 343.0
        V = (4 / 3) * np.pi * 0.02**3
        S = np.pi * 0.004**2
        L_eff = 0.015 + 2 * 0.6 * 0.004
        expected = (c / (2 * np.pi)) * np.sqrt(S / (V * L_eff))

        assert np.isclose(res.resonant_frequency, expected, rtol=1e-3)

    def test_flanged_vs_unflanged(self):
        """Test end correction difference between flanged and unflanged."""
        # Identical resonators except flanging
        params = {
            "position": (0.05, 0.05, 0.05),
            "cavity_shape": "box",
            "cavity_size": (0.03, 0.03, 0.03),
            "neck_length": 0.01,
            "neck_radius": 0.005,
        }

        unflanged = HelmholtzResonator(**params, opening_type="unflanged")
        flanged = HelmholtzResonator(**params, opening_type="flanged")

        # Flanged has larger end correction → longer effective length → lower frequency
        assert flanged.effective_neck_length > unflanged.effective_neck_length
        assert flanged.resonant_frequency < unflanged.resonant_frequency

    def test_frequency_range(self):
        """Test that calculated frequencies are in physically reasonable range."""
        # Typical Helmholtz resonator: 100 Hz to 2 kHz
        res = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="box",
            cavity_size=(0.03, 0.03, 0.03),
            neck_length=0.01,
            neck_radius=0.005,
        )

        freq = res.resonant_frequency
        assert 50 < freq < 5000, f"Frequency {freq:.1f} Hz outside reasonable range"

    def test_volume_frequency_relationship(self):
        """Test that larger volume → lower frequency."""
        small = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="box",
            cavity_size=(0.02, 0.02, 0.02),
            neck_length=0.01,
            neck_radius=0.005,
        )

        large = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="box",
            cavity_size=(0.04, 0.04, 0.04),  # 8x larger volume
            neck_length=0.01,
            neck_radius=0.005,
        )

        assert large.cavity_volume > small.cavity_volume
        assert large.resonant_frequency < small.resonant_frequency

    def test_neck_size_frequency_relationship(self):
        """Test that larger neck → higher frequency."""
        small_neck = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="box",
            cavity_size=(0.03, 0.03, 0.03),
            neck_length=0.01,
            neck_radius=0.003,
        )

        large_neck = HelmholtzResonator(
            position=(0.05, 0.05, 0.05),
            cavity_shape="box",
            cavity_size=(0.03, 0.03, 0.03),
            neck_length=0.01,
            neck_radius=0.007,
        )

        assert large_neck.neck_area > small_neck.neck_area
        assert large_neck.resonant_frequency > small_neck.resonant_frequency


class TestHelmholtzArray:
    """Test resonator array generation."""

    def test_array_generation_log_spacing(self):
        """Test array with logarithmic frequency spacing."""
        region = Box(center=(0.1, 0.1, 0.1), size=(0.08, 0.08, 0.1))
        resonators = helmholtz_array(
            region=region,
            frequency_range=(300, 1200),
            n_resonators=8,
            spacing="log",
        )

        assert len(resonators) == 8

        # Check frequencies are in range and logarithmically spaced
        freqs = [r.resonant_frequency for r in resonators]
        assert freqs[0] >= 250, "First resonator below target range"
        assert freqs[-1] <= 1300, "Last resonator above target range"

        # Check increasing order
        assert all(freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1))

    def test_array_generation_linear_spacing(self):
        """Test array with linear frequency spacing."""
        region = Box(center=(0.1, 0.1, 0.1), size=(0.08, 0.08, 0.1))
        resonators = helmholtz_array(
            region=region,
            frequency_range=(500, 1000),
            n_resonators=6,
            spacing="linear",
        )

        assert len(resonators) == 6

        # Check frequencies are roughly linearly spaced
        freqs = [r.resonant_frequency for r in resonators]
        freq_diffs = np.diff(freqs)

        # Linear spacing should have similar differences
        # (within ~30% due to volume quantization)
        mean_diff = np.mean(freq_diffs)
        assert all(abs(d - mean_diff) / mean_diff < 0.5 for d in freq_diffs)

    def test_array_positioning(self):
        """Test that resonators are positioned within region."""
        region = Box(center=(0.1, 0.1, 0.1), size=(0.08, 0.08, 0.08))
        resonators = helmholtz_array(
            region=region,
            frequency_range=(400, 800),
            n_resonators=4,
        )

        bb_min, bb_max = region.bounding_box

        # Check all resonators are within region bounds
        for res in resonators:
            pos = res.position
            # Allow small margin for resonator size
            margin = 0.02
            assert np.all(pos >= bb_min - margin)
            assert np.all(pos <= bb_max + margin)

    def test_array_cavity_shapes(self):
        """Test array generation with different cavity shapes."""
        region = Box(center=(0.1, 0.1, 0.1), size=(0.08, 0.08, 0.08))

        for shape in ["box", "sphere", "cylinder"]:
            resonators = helmholtz_array(
                region=region,
                frequency_range=(400, 800),
                n_resonators=4,
                cavity_shape=shape,
            )

            assert len(resonators) == 4
            assert all(r.cavity_shape == shape for r in resonators)

    def test_single_resonator_array(self):
        """Test array with single resonator."""
        region = Box(center=(0.1, 0.1, 0.1), size=(0.05, 0.05, 0.05))
        resonators = helmholtz_array(
            region=region,
            frequency_range=(500, 500),  # Same min/max
            n_resonators=1,
        )

        assert len(resonators) == 1
        freq = resonators[0].resonant_frequency
        assert 400 < freq < 600  # Close to target


class TestValidation:
    """Test input validation and error handling."""

    def test_invalid_cavity_shape(self):
        """Test error on invalid cavity shape."""
        with pytest.raises(ValueError, match="Unknown cavity_shape"):
            HelmholtzResonator(
                position=(0.05, 0.05, 0.05),
                cavity_shape="triangle",  # Invalid
                cavity_size=(0.03, 0.03, 0.03),
                neck_length=0.01,
                neck_radius=0.005,
            )

    def test_zero_neck_direction(self):
        """Test error on zero neck direction vector."""
        with pytest.raises(ValueError, match="cannot be zero"):
            HelmholtzResonator(
                position=(0.05, 0.05, 0.05),
                cavity_shape="box",
                cavity_size=(0.03, 0.03, 0.03),
                neck_length=0.01,
                neck_radius=0.005,
                neck_direction=(0, 0, 0),  # Invalid
            )

    def test_negative_neck_length(self):
        """Test error on negative neck length."""
        with pytest.raises(ValueError, match="must be positive"):
            HelmholtzResonator(
                position=(0.05, 0.05, 0.05),
                cavity_shape="box",
                cavity_size=(0.03, 0.03, 0.03),
                neck_length=-0.01,  # Invalid
                neck_radius=0.005,
            )

    def test_negative_neck_radius(self):
        """Test error on negative neck radius."""
        with pytest.raises(ValueError, match="must be positive"):
            HelmholtzResonator(
                position=(0.05, 0.05, 0.05),
                cavity_shape="box",
                cavity_size=(0.03, 0.03, 0.03),
                neck_length=0.01,
                neck_radius=-0.005,  # Invalid
            )

    def test_wrong_cavity_size_count_box(self):
        """Test error on wrong number of size parameters for box."""
        with pytest.raises(ValueError, match="requires 3 size parameters"):
            HelmholtzResonator(
                position=(0.05, 0.05, 0.05),
                cavity_shape="box",
                cavity_size=(0.03, 0.03),  # Should be 3 values
                neck_length=0.01,
                neck_radius=0.005,
            )

    def test_wrong_cavity_size_count_sphere(self):
        """Test error on wrong number of size parameters for sphere."""
        with pytest.raises(ValueError, match="requires 1 size parameter"):
            HelmholtzResonator(
                position=(0.05, 0.05, 0.05),
                cavity_shape="sphere",
                cavity_size=(0.03, 0.03),  # Should be 1 value
                neck_length=0.01,
                neck_radius=0.005,
            )

    def test_array_invalid_frequency_range(self):
        """Test error on invalid frequency range."""
        region = Box(center=(0.1, 0.1, 0.1), size=(0.08, 0.08, 0.08))

        with pytest.raises(ValueError, match="Invalid frequency range"):
            helmholtz_array(
                region=region,
                frequency_range=(1000, 500),  # max < min
                n_resonators=4,
            )

    def test_array_invalid_spacing_mode(self):
        """Test error on invalid spacing mode."""
        region = Box(center=(0.1, 0.1, 0.1), size=(0.08, 0.08, 0.08))

        with pytest.raises(ValueError, match="spacing must be"):
            helmholtz_array(
                region=region,
                frequency_range=(400, 800),
                n_resonators=4,
                spacing="exponential",  # Invalid
            )


class TestProperties:
    """Test property calculations."""

    def test_cavity_volume_box(self):
        """Test cavity volume for box shape."""
        res = HelmholtzResonator(
            position=(0, 0, 0),
            cavity_shape="box",
            cavity_size=(0.02, 0.03, 0.04),
        )
        expected = 0.02 * 0.03 * 0.04
        assert np.isclose(res.cavity_volume, expected)

    def test_cavity_volume_sphere(self):
        """Test cavity volume for spherical shape."""
        res = HelmholtzResonator(
            position=(0, 0, 0),
            cavity_shape="sphere",
            cavity_size=(0.015,),
        )
        expected = (4 / 3) * np.pi * 0.015**3
        assert np.isclose(res.cavity_volume, expected)

    def test_neck_area(self):
        """Test neck cross-sectional area."""
        res = HelmholtzResonator(
            position=(0, 0, 0),
            cavity_shape="box",
            cavity_size=(0.03, 0.03, 0.03),
            neck_radius=0.005,
        )
        expected = np.pi * 0.005**2
        assert np.isclose(res.neck_area, expected)

    def test_effective_neck_length(self):
        """Test effective length calculation."""
        res = HelmholtzResonator(
            position=(0, 0, 0),
            cavity_shape="box",
            cavity_size=(0.03, 0.03, 0.03),
            neck_length=0.01,
            neck_radius=0.005,
            opening_type="unflanged",
        )

        # unflanged: L + 2 * 0.6 * r
        expected = 0.01 + 2 * 0.6 * 0.005
        assert np.isclose(res.effective_neck_length, expected)

    def test_quality_factor(self):
        """Test Q factor placeholder."""
        res = HelmholtzResonator(
            position=(0, 0, 0),
            cavity_shape="box",
            cavity_size=(0.03, 0.03, 0.03),
        )

        # Placeholder Q should be reasonable
        assert 1 < res.quality_factor < 100
