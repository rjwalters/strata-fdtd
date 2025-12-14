"""
Unit tests for loudspeaker enclosure builder API.

Tests verify:
- LoudspeakerEnclosure construction
- Component addition (drivers, ports, chambers, bracing)
- CSG geometry building
- Factory functions (tower_speaker, bookshelf_speaker)
- Material region specification
"""

import numpy as np

from strata_fdtd import LoudspeakerEnclosure, UniformGrid, bookshelf_speaker, tower_speaker
from strata_fdtd.resonator import HelmholtzResonator


class TestLoudspeakerEnclosure:
    """Tests for LoudspeakerEnclosure class."""

    def test_construction(self):
        """Test basic enclosure construction."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))

        assert enc.external_size == (0.2, 0.25, 0.4)
        assert enc.wall_thickness == 0.019  # Default 19mm

    def test_custom_wall_thickness(self):
        """Test construction with custom wall thickness."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4), wall_thickness=0.025)

        assert enc.wall_thickness == 0.025

    def test_add_driver_front_face(self):
        """Test adding driver to front baffle."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_driver((0.1, 0.3), 0.025, baffle_face="front", name="tweeter")

        assert len(enc._drivers) == 1
        assert enc._drivers[0]["position"] == (0.1, 0.3)
        assert enc._drivers[0]["diameter"] == 0.025
        assert enc._drivers[0]["face"] == "front"
        assert enc._drivers[0]["name"] == "tweeter"

    def test_add_driver_default_name(self):
        """Test driver gets auto-generated name if not provided."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_driver((0.1, 0.3), 0.025)

        assert enc._drivers[0]["name"] == "driver_0"

    def test_add_driver_method_chaining(self):
        """Test that add_driver returns self for chaining."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        result = enc.add_driver((0.1, 0.3), 0.025)

        assert result is enc

    def test_add_multiple_drivers(self):
        """Test adding multiple drivers."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_driver((0.1, 0.3), 0.025, name="tweeter")
        enc.add_driver((0.1, 0.15), 0.065, name="woofer")

        assert len(enc._drivers) == 2
        assert enc._drivers[0]["name"] == "tweeter"
        assert enc._drivers[1]["name"] == "woofer"

    def test_add_port(self):
        """Test adding bass reflex port."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_port((0.1, 0.1), 0.05, 0.15, baffle_face="front")

        assert len(enc._ports) == 1
        assert enc._ports[0]["position"] == (0.1, 0.1)
        assert enc._ports[0]["diameter"] == 0.05
        assert enc._ports[0]["length"] == 0.15

    def test_add_flared_port(self):
        """Test adding flared port."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_port((0.1, 0.1), 0.05, 0.15, flare_ratio=1.5, flare_length=0.03)

        assert enc._ports[0]["flare_ratio"] == 1.5
        assert enc._ports[0]["flare_length"] == 0.03

    def test_add_chamber_divider(self):
        """Test adding chamber divider."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_chamber_divider(z_position=0.2)

        assert len(enc._chambers) == 1
        assert enc._chambers[0]["z"] == 0.2
        assert enc._chambers[0]["thickness"] == 0.019

    def test_add_chamber_divider_with_holes(self):
        """Test adding chamber divider with holes."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        holes = [(0.1, 0.125, 0.01)]
        enc.add_chamber_divider(z_position=0.2, hole_positions=holes)

        assert enc._chambers[0]["holes"] == holes

    def test_add_brace(self):
        """Test adding internal brace."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_brace((0.02, 0.125, 0.2), (0.18, 0.125, 0.2), 0.02, 0.015)

        assert len(enc._bracing) == 1
        assert enc._bracing[0]["start"] == (0.02, 0.125, 0.2)
        assert enc._bracing[0]["end"] == (0.18, 0.125, 0.2)
        assert enc._bracing[0]["width"] == 0.02

    def test_add_absorber_region(self):
        """Test adding absorber material region."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        bounds = ((0.02, 0.02, 0.02), (0.18, 0.23, 0.1))
        enc.add_absorber_region(bounds, material_id=1)

        assert len(enc._absorbers) == 1
        assert enc._absorbers[0]["bounds"] == bounds
        assert enc._absorbers[0]["material_id"] == 1

    def test_add_helmholtz_array(self):
        """Test adding Helmholtz resonator array."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        bounds = ((0.02, 0.02, 0.02), (0.18, 0.23, 0.2))
        enc.add_helmholtz_array(bounds, (300, 1200), 8)

        assert len(enc._resonators) == 1
        assert enc._resonators[0]["bounds"] == bounds
        assert enc._resonators[0]["freq_range"] == (300, 1200)
        assert enc._resonators[0]["n"] == 8

    def test_build_empty_enclosure(self):
        """Test building enclosure with no drivers or ports."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        geometry = enc.build()

        # Should return a valid SDF primitive
        assert geometry is not None
        assert hasattr(geometry, 'sdf')
        assert hasattr(geometry, 'bounding_box')

    def test_build_with_driver(self):
        """Test building enclosure with driver cutout."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_driver((0.1, 0.3), 0.025, baffle_face="front")
        geometry = enc.build()

        assert geometry is not None

    def test_build_with_port(self):
        """Test building enclosure with port."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_port((0.1, 0.1), 0.05, 0.15, baffle_face="front")
        geometry = enc.build()

        assert geometry is not None

    def test_build_with_chamber(self):
        """Test building enclosure with chamber divider."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_chamber_divider(z_position=0.2)
        geometry = enc.build()

        assert geometry is not None

    def test_build_complete_enclosure(self):
        """Test building complete enclosure with all components."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_driver((0.1, 0.3), 0.025, baffle_face="front", name="tweeter")
        enc.add_driver((0.1, 0.15), 0.065, baffle_face="front", name="woofer")
        enc.add_port((0.1, 0.05), 0.05, 0.15, baffle_face="front")
        enc.add_chamber_divider(z_position=0.2)
        enc.add_brace((0.02, 0.125, 0.2), (0.18, 0.125, 0.2), 0.02, 0.015)

        geometry = enc.build()

        assert geometry is not None

    def test_voxelize_enclosure(self):
        """Test voxelizing enclosure to grid."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_driver((0.1, 0.3), 0.025, baffle_face="front")
        geometry = enc.build()

        grid = UniformGrid(shape=(40, 50, 80), resolution=5e-3)
        mask = geometry.voxelize(grid)

        # Check that voxelization produces reasonable output
        assert mask.shape == (40, 50, 80)
        assert mask.dtype == bool
        # Should have both solid (False) and air (True) voxels
        assert np.any(mask)
        assert np.any(~mask)

    def test_get_material_regions_empty(self):
        """Test getting material regions when none defined."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        regions = enc.get_material_regions()

        assert regions == []

    def test_get_material_regions(self):
        """Test getting material regions."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        bounds = ((0.02, 0.02, 0.02), (0.18, 0.23, 0.1))
        enc.add_absorber_region(bounds, material_id=1)

        regions = enc.get_material_regions()

        assert len(regions) == 1
        primitive, material_id = regions[0]
        assert material_id == 1
        assert hasattr(primitive, 'sdf')

    def test_fluent_api(self):
        """Test fluent API method chaining."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))

        # Should be able to chain calls
        result = (enc
                  .add_driver((0.1, 0.3), 0.025)
                  .add_driver((0.1, 0.15), 0.065)
                  .add_port((0.1, 0.05), 0.05, 0.15)
                  .add_chamber_divider(0.2))

        assert result is enc
        assert len(enc._drivers) == 2
        assert len(enc._ports) == 1
        assert len(enc._chambers) == 1


class TestTowerSpeaker:
    """Tests for tower_speaker factory function."""

    def test_construction(self):
        """Test tower speaker construction."""
        enc = tower_speaker(
            width=0.2,
            depth=0.3,
            height=1.0,
            woofer_diameter=0.2,
            midrange_diameter=0.13,
            tweeter_diameter=0.025
        )

        assert isinstance(enc, LoudspeakerEnclosure)
        assert enc.external_size == (0.2, 0.3, 1.0)

    def test_has_three_drivers(self):
        """Test tower speaker has three drivers."""
        enc = tower_speaker(
            width=0.2,
            depth=0.3,
            height=1.0,
            woofer_diameter=0.2,
            midrange_diameter=0.13,
            tweeter_diameter=0.025
        )

        assert len(enc._drivers) == 3
        # Check driver names
        names = [d["name"] for d in enc._drivers]
        assert "tweeter" in names
        assert "midrange" in names
        assert "woofer" in names

    def test_has_chamber_dividers(self):
        """Test tower speaker has chamber dividers."""
        enc = tower_speaker(
            width=0.2,
            depth=0.3,
            height=1.0,
            woofer_diameter=0.2,
            midrange_diameter=0.13,
            tweeter_diameter=0.025
        )

        # Should have 2 chamber dividers
        assert len(enc._chambers) == 2

    def test_can_build(self):
        """Test tower speaker can be built."""
        enc = tower_speaker(
            width=0.2,
            depth=0.3,
            height=1.0,
            woofer_diameter=0.2,
            midrange_diameter=0.13,
            tweeter_diameter=0.025
        )

        geometry = enc.build()
        assert geometry is not None


class TestBookshelfSpeaker:
    """Tests for bookshelf_speaker factory function."""

    def test_construction(self):
        """Test bookshelf speaker construction."""
        enc = bookshelf_speaker(
            width=0.2,
            depth=0.25,
            height=0.35,
            woofer_diameter=0.13,
            tweeter_diameter=0.025
        )

        assert isinstance(enc, LoudspeakerEnclosure)
        assert enc.external_size == (0.2, 0.25, 0.35)

    def test_has_two_drivers(self):
        """Test bookshelf speaker has two drivers."""
        enc = bookshelf_speaker(
            width=0.2,
            depth=0.25,
            height=0.35,
            woofer_diameter=0.13,
            tweeter_diameter=0.025
        )

        assert len(enc._drivers) == 2
        # Check driver names
        names = [d["name"] for d in enc._drivers]
        assert "tweeter" in names
        assert "woofer" in names

    def test_has_port(self):
        """Test bookshelf speaker has bass reflex port."""
        enc = bookshelf_speaker(
            width=0.2,
            depth=0.25,
            height=0.35,
            woofer_diameter=0.13,
            tweeter_diameter=0.025
        )

        assert len(enc._ports) == 1

    def test_custom_port_parameters(self):
        """Test bookshelf speaker with custom port parameters."""
        enc = bookshelf_speaker(
            width=0.2,
            depth=0.25,
            height=0.35,
            woofer_diameter=0.13,
            tweeter_diameter=0.025,
            port_diameter=0.06,
            port_length=0.18
        )

        assert enc._ports[0]["diameter"] == 0.06
        assert enc._ports[0]["length"] == 0.18

    def test_can_build(self):
        """Test bookshelf speaker can be built."""
        enc = bookshelf_speaker(
            width=0.2,
            depth=0.25,
            height=0.35,
            woofer_diameter=0.13,
            tweeter_diameter=0.025
        )

        geometry = enc.build()
        assert geometry is not None


class TestHelmholtzResonatorIntegration:
    """Tests for Helmholtz resonator array integration."""

    def test_build_helmholtz_array_method(self):
        """Test _build_helmholtz_array method."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))

        # Create spec matching add_helmholtz_array format
        spec = {
            "bounds": ((0.02, 0.02, 0.02), (0.18, 0.23, 0.2)),
            "freq_range": (300, 1200),
            "n": 6,
        }

        resonators = enc._build_helmholtz_array(spec)

        # Check we got the right number of resonators
        assert len(resonators) == 6

        # Check all are HelmholtzResonator instances
        assert all(isinstance(r, HelmholtzResonator) for r in resonators)

        # Check frequencies span the requested range (with some tolerance)
        freqs = [r.resonant_frequency for r in resonators]
        assert min(freqs) >= 250  # Allow 50Hz margin
        assert max(freqs) <= 1300

    def test_build_with_helmholtz_array(self):
        """Test building enclosure with Helmholtz resonator array."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_driver((0.1, 0.3), 0.025, baffle_face="front")
        enc.add_helmholtz_array(
            region_bounds=((0.02, 0.02, 0.02), (0.18, 0.23, 0.2)),
            frequency_range=(300, 1200),
            n_resonators=8,
        )

        geometry = enc.build()

        # Should successfully build without errors
        assert geometry is not None
        assert hasattr(geometry, 'sdf')
        assert hasattr(geometry, 'bounding_box')

    def test_voxelize_with_helmholtz_array(self):
        """Test voxelizing enclosure with Helmholtz resonators."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))
        enc.add_helmholtz_array(
            region_bounds=((0.02, 0.02, 0.02), (0.18, 0.23, 0.2)),
            frequency_range=(400, 800),
            n_resonators=4,
        )

        geometry = enc.build()
        grid = UniformGrid(shape=(40, 50, 80), resolution=5e-3)
        mask = geometry.voxelize(grid)

        # Check voxelization produces valid output
        assert mask.shape == (40, 50, 80)
        assert mask.dtype == bool
        # Should have both solid and air voxels
        assert np.any(mask)
        assert np.any(~mask)

    def test_multiple_helmholtz_arrays(self):
        """Test building with multiple Helmholtz arrays."""
        enc = LoudspeakerEnclosure(external_size=(0.3, 0.3, 0.6))

        # Add two different arrays in different regions
        enc.add_helmholtz_array(
            region_bounds=((0.02, 0.02, 0.1), (0.28, 0.28, 0.25)),
            frequency_range=(200, 500),
            n_resonators=4,
        )
        enc.add_helmholtz_array(
            region_bounds=((0.02, 0.02, 0.35), (0.28, 0.28, 0.5)),
            frequency_range=(600, 1500),
            n_resonators=6,
        )

        geometry = enc.build()
        assert geometry is not None

        # Check both arrays were stored
        assert len(enc._resonators) == 2

    def test_helmholtz_array_with_driver_and_port(self):
        """Test complete enclosure with drivers, ports, and Helmholtz array."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))

        # Add standard components
        enc.add_driver((0.1, 0.3), 0.025, baffle_face="front", name="tweeter")
        enc.add_driver((0.1, 0.15), 0.065, baffle_face="front", name="woofer")
        enc.add_port((0.1, 0.05), 0.05, 0.15, baffle_face="front")

        # Add Helmholtz array for absorption
        enc.add_helmholtz_array(
            region_bounds=((0.02, 0.02, 0.02), (0.18, 0.08, 0.12)),
            frequency_range=(300, 1200),
            n_resonators=8,
        )

        geometry = enc.build()
        assert geometry is not None

    def test_resonator_positioning_within_bounds(self):
        """Test that resonators are positioned within specified bounds."""
        enc = LoudspeakerEnclosure(external_size=(0.3, 0.3, 0.3))

        spec = {
            "bounds": ((0.05, 0.05, 0.05), (0.25, 0.25, 0.25)),
            "freq_range": (400, 800),
            "n": 8,
        }

        resonators = enc._build_helmholtz_array(spec)

        # Extract positions and check they're within bounds (with margin for cavity size)
        min_corner = np.array([0.05, 0.05, 0.05])
        max_corner = np.array([0.25, 0.25, 0.25])

        for res in resonators:
            pos = np.array(res.position)
            # Allow margin for cavity size (~0.03m)
            margin = 0.04
            assert np.all(pos >= min_corner - margin)
            assert np.all(pos <= max_corner + margin)

    def test_resonator_frequency_distribution(self):
        """Test that resonator frequencies are properly distributed."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))

        spec = {
            "bounds": ((0.02, 0.02, 0.02), (0.18, 0.23, 0.2)),
            "freq_range": (500, 1000),
            "n": 5,
        }

        resonators = enc._build_helmholtz_array(spec)
        freqs = [r.resonant_frequency for r in resonators]

        # Check frequencies are in ascending order (log spacing)
        assert all(freqs[i] < freqs[i + 1] for i in range(len(freqs) - 1))

        # Check coverage of frequency range
        freq_span = max(freqs) - min(freqs)
        expected_span = 1000 - 500
        # Allow 40% tolerance due to volume quantization
        assert freq_span > expected_span * 0.6

    def test_single_helmholtz_resonator(self):
        """Test array with single resonator."""
        enc = LoudspeakerEnclosure(external_size=(0.2, 0.25, 0.4))

        spec = {
            "bounds": ((0.05, 0.05, 0.05), (0.15, 0.15, 0.15)),
            "freq_range": (600, 600),  # Single frequency
            "n": 1,
        }

        resonators = enc._build_helmholtz_array(spec)

        assert len(resonators) == 1
        # Should be close to target frequency
        assert 500 < resonators[0].resonant_frequency < 700
