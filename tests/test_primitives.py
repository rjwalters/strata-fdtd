"""Tests for primitive geometry building blocks."""

import numpy as np
import pytest

from strata_fdtd.primitives import (
    SPEED_OF_SOUND,
    Channel,
    HelmholtzCell,
    SerpentineChannel,
    SphereCell,
    SphereLattice,
    TaperedChannel,
    connect_points,
)


class TestHelmholtzCell:
    """Tests for the HelmholtzCell class."""

    def test_create_basic_cell(self):
        """Test creating a basic Helmholtz cell."""
        cell = HelmholtzCell(
            cavity_x=20.0,
            cavity_y=20.0,
            cavity_z=2,
            neck_x=5.0,
            neck_y=5.0,
            neck_z=1,
            position=(0.0, 0.0, 0),
        )

        assert cell.cavity_x == 20.0
        assert cell.neck_z == 1
        assert cell.total_height_slices() == 3

    def test_cavity_volume_calculation(self):
        """Test cavity volume is calculated correctly."""
        cell = HelmholtzCell(
            cavity_x=20.0,  # 20mm
            cavity_y=30.0,  # 30mm
            cavity_z=2,     # 2 slices = 12mm
            neck_x=5.0,
            neck_y=5.0,
            neck_z=1,
            position=(0.0, 0.0, 0),
        )

        # Volume = 20 * 30 * 12 = 7200 mm³
        assert cell.cavity_volume_mm3 == pytest.approx(7200.0)
        # In m³: 7200e-9 = 7.2e-6
        assert cell.cavity_volume_m3 == pytest.approx(7.2e-6)

    def test_neck_area_calculation(self):
        """Test neck area is calculated correctly."""
        cell = HelmholtzCell(
            cavity_x=20.0,
            cavity_y=20.0,
            cavity_z=2,
            neck_x=5.0,  # 5mm
            neck_y=8.0,  # 8mm
            neck_z=1,
            position=(0.0, 0.0, 0),
        )

        # Area = 5 * 8 = 40 mm²
        assert cell.neck_area_mm2 == pytest.approx(40.0)
        # In m²: 40e-6
        assert cell.neck_area_m2 == pytest.approx(40e-6)

    def test_resonant_frequency_formula(self):
        """Test resonant frequency calculation matches analytical formula."""
        # Design a cell with known resonant frequency
        # Using f = (c / 2π) * sqrt(S / (V * L_eff))
        cell = HelmholtzCell(
            cavity_x=30.0,
            cavity_y=30.0,
            cavity_z=3,  # 18mm deep cavity
            neck_x=6.0,
            neck_y=6.0,
            neck_z=1,  # 6mm neck length
            position=(0.0, 0.0, 0),
        )

        # Calculate expected frequency manually
        c = SPEED_OF_SOUND  # 343 m/s
        V = 30e-3 * 30e-3 * 18e-3  # m³
        S = 6e-3 * 6e-3  # m²
        L = 6e-3  # m
        L_eff = L + 1.7 * np.sqrt(S / np.pi)
        expected_f = (c / (2 * np.pi)) * np.sqrt(S / (V * L_eff))

        # Allow 5% tolerance as mentioned in issue
        assert cell.resonant_frequency() == pytest.approx(expected_f, rel=0.05)

    def test_resonant_frequency_range(self):
        """Test that designed cells produce frequencies in expected range."""
        # Cell designed for ~200 Hz (large cavity, long neck)
        cell_low = HelmholtzCell(
            cavity_x=50.0,
            cavity_y=50.0,
            cavity_z=5,  # Large cavity: 30mm height
            neck_x=10.0,
            neck_y=10.0,
            neck_z=2,    # Long neck: 12mm
            position=(0.0, 0.0, 0),
        )

        # Cell designed for higher frequency (smaller cavity)
        cell_high = HelmholtzCell(
            cavity_x=20.0,
            cavity_y=20.0,
            cavity_z=1,  # Small cavity: 6mm height
            neck_x=8.0,
            neck_y=8.0,
            neck_z=1,    # Short neck: 6mm
            position=(0.0, 0.0, 0),
        )

        f_low = cell_low.resonant_frequency()
        f_high = cell_high.resonant_frequency()

        # Verify ordering and reasonable range
        # Small cells can produce frequencies up to several kHz
        assert 50 < f_low < 500  # Low frequency cell
        assert 500 < f_high < 5000  # Higher frequency cell (small cavity = high freq)
        assert f_low < f_high

    def test_to_stack_generates_correct_slices(self):
        """Test that to_stack creates the right number of slices."""
        cell = HelmholtzCell(
            cavity_x=20.0,
            cavity_y=20.0,
            cavity_z=2,  # 2 cavity slices
            neck_x=5.0,
            neck_y=5.0,
            neck_z=1,   # 1 neck slice
            position=(0.0, 0.0, 0),
        )

        stack = cell.to_stack(resolution=1.0)

        assert stack.num_slices == 3  # cavity_z + neck_z

    def test_check_manufacturable_valid_cell(self):
        """Test that valid cells pass manufacturing checks."""
        cell = HelmholtzCell(
            cavity_x=20.0,
            cavity_y=20.0,
            cavity_z=2,
            neck_x=5.0,  # Above min_gap (2mm)
            neck_y=5.0,
            neck_z=1,
            position=(0.0, 0.0, 0),
        )

        violations = cell.check_manufacturable(min_gap=2.0)
        assert len(violations) == 0

    def test_check_manufacturable_narrow_neck(self):
        """Test that cells with narrow necks are flagged."""
        cell = HelmholtzCell(
            cavity_x=20.0,
            cavity_y=20.0,
            cavity_z=2,
            neck_x=1.5,  # Below min_gap (2mm)
            neck_y=5.0,
            neck_z=1,
            position=(0.0, 0.0, 0),
        )

        violations = cell.check_manufacturable(min_gap=2.0)
        assert len(violations) > 0
        assert any(v.constraint == "min_gap_width" for v in violations)

    def test_invalid_neck_direction_raises(self):
        """Test that invalid neck direction raises error."""
        with pytest.raises(ValueError, match="neck_direction"):
            HelmholtzCell(
                cavity_x=20.0,
                cavity_y=20.0,
                cavity_z=2,
                neck_x=5.0,
                neck_y=5.0,
                neck_z=1,
                position=(0.0, 0.0, 0),
                neck_direction="invalid",
            )


class TestChannel:
    """Tests for the Channel class."""

    def test_create_straight_channel(self):
        """Test creating a straight channel."""
        channel = Channel(
            width=5.0,
            height=1,
            path=[(0.0, 0.0), (100.0, 0.0)],
            z_start=0,
        )

        assert channel.width == 5.0
        assert channel.total_length() == pytest.approx(100.0)

    def test_channel_with_turns(self):
        """Test channel with multiple waypoints."""
        # L-shaped path
        channel = Channel(
            width=5.0,
            height=1,
            path=[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)],
            z_start=0,
        )

        # Length should be 50 + 50 = 100
        assert channel.total_length() == pytest.approx(100.0)

    def test_channel_to_stack_creates_slices(self):
        """Test that to_stack creates geometry."""
        channel = Channel(
            width=5.0,
            height=2,
            path=[(10.0, 10.0), (50.0, 10.0)],
            z_start=0,
        )

        stack = channel.to_stack(resolution=1.0)

        assert stack.num_slices == 2
        # Check that there's air along the channel
        slice_0 = stack.get_slice(0)
        assert slice_0 is not None

    def test_channel_requires_two_waypoints(self):
        """Test that channel path needs at least 2 points."""
        with pytest.raises(ValueError, match="at least 2 waypoints"):
            Channel(
                width=5.0,
                height=1,
                path=[(0.0, 0.0)],
                z_start=0,
            )


class TestTaperedChannel:
    """Tests for the TaperedChannel class."""

    def test_create_tapered_channel(self):
        """Test creating a tapered channel."""
        channel = TaperedChannel(
            width_start=5.0,
            width_end=10.0,
            height=1,
            path=[(0.0, 0.0), (100.0, 0.0)],
            z_start=0,
        )

        assert channel.width_start == 5.0
        assert channel.width_end == 10.0

    def test_width_at_position_interpolates(self):
        """Test that width interpolates linearly."""
        channel = TaperedChannel(
            width_start=10.0,
            width_end=20.0,
            height=1,
            path=[(0.0, 0.0), (100.0, 0.0)],
            z_start=0,
        )

        assert channel.width_at_position(0.0) == pytest.approx(10.0)
        assert channel.width_at_position(0.5) == pytest.approx(15.0)
        assert channel.width_at_position(1.0) == pytest.approx(20.0)

    def test_tapered_channel_to_stack(self):
        """Test that tapered channel generates geometry."""
        channel = TaperedChannel(
            width_start=5.0,
            width_end=10.0,
            height=1,
            path=[(10.0, 10.0), (50.0, 10.0)],
            z_start=0,
        )

        stack = channel.to_stack(resolution=1.0)
        assert stack.num_slices == 1


class TestSerpentineChannel:
    """Tests for the SerpentineChannel class."""

    def test_create_serpentine_channel(self):
        """Test creating a serpentine channel."""
        channel = SerpentineChannel(
            width=5.0,
            height=1,
            fold_count=3,
            fold_spacing=10.0,
            total_length=100.0,
            z_start=0,
        )

        assert channel.fold_count == 3
        assert channel.total_length == 100.0

    def test_serpentine_actual_length(self):
        """Test that serpentine achieves approximately target length."""
        channel = SerpentineChannel(
            width=5.0,
            height=1,
            fold_count=3,
            fold_spacing=10.0,
            total_length=100.0,
            z_start=0,
        )

        # Actual length should be close to total_length
        # (within 1% as mentioned in issue)
        actual = channel.actual_length()
        assert actual == pytest.approx(100.0, rel=0.01)

    def test_serpentine_to_stack(self):
        """Test serpentine channel generates valid geometry."""
        channel = SerpentineChannel(
            width=5.0,
            height=1,
            fold_count=2,
            fold_spacing=15.0,
            total_length=80.0,
            z_start=0,
        )

        stack = channel.to_stack(resolution=1.0)
        assert stack.num_slices == 1

    def test_serpentine_invalid_fold_count(self):
        """Test that fold_count must be at least 1."""
        with pytest.raises(ValueError, match="fold_count"):
            SerpentineChannel(
                width=5.0,
                height=1,
                fold_count=0,
                fold_spacing=10.0,
                total_length=50.0,
                z_start=0,
            )

    def test_serpentine_different_directions(self):
        """Test serpentine in different initial directions."""
        for direction in ["+x", "-x", "+y", "-y"]:
            channel = SerpentineChannel(
                width=5.0,
                height=1,
                fold_count=2,
                fold_spacing=10.0,
                total_length=60.0,
                z_start=0,
                direction=direction,
            )
            stack = channel.to_stack()
            assert stack.num_slices == 1


class TestConnectPoints:
    """Tests for the connect_points function."""

    def test_connect_two_points_same_level(self):
        """Test connecting two points at the same z level."""
        points = [(10.0, 10.0, 0), (50.0, 10.0, 0)]
        stack = connect_points(points, width=5.0)

        assert stack.num_slices >= 1
        # Should have a channel connecting the points
        slice_0 = stack.get_slice(0)
        assert slice_0 is not None

    def test_connect_insufficient_points_returns_empty(self):
        """Test that fewer than 2 points returns empty stack."""
        points = [(10.0, 10.0, 0)]
        stack = connect_points(points, width=5.0)

        assert stack.num_slices == 0


class TestSphereCell:
    """Tests for the SphereCell class."""

    def test_create_basic_sphere(self):
        """Test creating a basic sphere cell."""
        sphere = SphereCell(
            diameter=30.0,
            position=(0.0, 0.0, 5),
        )

        assert sphere.diameter == 30.0
        assert sphere.radius == 15.0
        assert sphere.position == (0.0, 0.0, 5)

    def test_volume_calculation(self):
        """Test sphere volume calculation."""
        sphere = SphereCell(diameter=30.0, position=(0.0, 0.0, 0))

        # Volume = (4/3) * π * r³ = (4/3) * π * 15³
        expected_mm3 = (4 / 3) * np.pi * 15**3
        assert sphere.volume_mm3 == pytest.approx(expected_mm3)

        # In m³
        expected_m3 = expected_mm3 * 1e-9
        assert sphere.volume_m3 == pytest.approx(expected_m3)

    def test_num_slices_calculation(self):
        """Test that num_slices correctly spans the diameter."""
        # 30mm sphere with 6mm slices = 5 slices
        sphere = SphereCell(diameter=30.0, position=(0.0, 0.0, 0))
        assert sphere.num_slices() == 5

        # 12mm sphere = 2 slices
        sphere2 = SphereCell(diameter=12.0, position=(0.0, 0.0, 0))
        assert sphere2.num_slices() == 2

        # Small sphere (< slice thickness) = 1 slice minimum
        sphere3 = SphereCell(diameter=5.0, position=(0.0, 0.0, 0))
        assert sphere3.num_slices() == 1

    def test_radius_at_slice_center(self):
        """Test radius at center slice equals full radius."""
        sphere = SphereCell(diameter=30.0, position=(0.0, 0.0, 0))

        # At center (offset=0), radius should be full
        assert sphere.radius_at_slice(0) == pytest.approx(15.0)

    def test_radius_at_slice_off_center(self):
        """Test radius decreases away from center."""
        sphere = SphereCell(diameter=30.0, position=(0.0, 0.0, 0))

        # At offset=1 (6mm from center), r = sqrt(15² - 6²) = sqrt(189)
        expected = np.sqrt(15**2 - 6**2)
        assert sphere.radius_at_slice(1) == pytest.approx(expected)
        assert sphere.radius_at_slice(-1) == pytest.approx(expected)

        # Radius should be symmetric
        assert sphere.radius_at_slice(1) == sphere.radius_at_slice(-1)

    def test_radius_outside_sphere_is_zero(self):
        """Test radius is zero outside sphere bounds."""
        sphere = SphereCell(diameter=12.0, position=(0.0, 0.0, 0))

        # 12mm diameter, 6mm radius, with 6mm slices
        # At offset=2 (12mm from center), should be outside
        assert sphere.radius_at_slice(3) == 0.0
        assert sphere.radius_at_slice(-3) == 0.0

    def test_to_stack_generates_correct_slices(self):
        """Test that to_stack creates geometry with correct slice count."""
        sphere = SphereCell(diameter=30.0, position=(0.0, 0.0, 5))

        stack = sphere.to_stack(resolution=1.0)

        # 30mm / 6mm = 5 slices
        assert stack.num_slices == 5

    def test_to_stack_creates_circular_cuts(self):
        """Test that slice masks contain circular air regions."""
        sphere = SphereCell(diameter=30.0, position=(0.0, 0.0, 5))

        stack = sphere.to_stack(resolution=1.0)

        # Center slice should have the largest air region
        center_slice = stack.get_slice(5)
        assert center_slice is not None
        assert center_slice.mask.any()  # Has some air

        # Edge slices should have smaller air regions
        first_slice = stack.slices[0]
        last_slice = stack.slices[-1]

        center_area = center_slice.mask.sum()
        first_area = first_slice.mask.sum()
        last_area = last_slice.mask.sum()

        # Center should have larger area than edges
        assert center_area > first_area
        assert center_area > last_area

    def test_check_manufacturable_valid_sphere(self):
        """Test that valid spheres pass manufacturing checks."""
        sphere = SphereCell(diameter=30.0, position=(0.0, 0.0, 5))

        violations = sphere.check_manufacturable(min_gap=2.0)
        assert len(violations) == 0

    def test_check_manufacturable_too_small_diameter(self):
        """Test that spheres smaller than min_gap*2 are flagged."""
        sphere = SphereCell(diameter=3.0, position=(0.0, 0.0, 0))

        violations = sphere.check_manufacturable(min_gap=2.0)
        assert any(v.constraint == "min_sphere_diameter" for v in violations)

    def test_check_manufacturable_too_few_slices(self):
        """Test that spheres with < 2 slices are flagged."""
        sphere = SphereCell(diameter=5.0, position=(0.0, 0.0, 0))

        violations = sphere.check_manufacturable(min_gap=2.0)
        assert any(v.constraint == "min_sphere_slices" for v in violations)

    def test_invalid_diameter_raises(self):
        """Test that non-positive diameter raises error."""
        with pytest.raises(ValueError, match="positive"):
            SphereCell(diameter=0.0, position=(0.0, 0.0, 0))

        with pytest.raises(ValueError, match="positive"):
            SphereCell(diameter=-10.0, position=(0.0, 0.0, 0))


class TestSphereLattice:
    """Tests for the SphereLattice class."""

    def test_create_cubic_lattice(self):
        """Test creating a cubic lattice."""
        lattice = SphereLattice(
            sphere_diameter=20.0,
            tube_diameter=4.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
        )

        assert lattice.lattice_type == "cubic"
        assert lattice.packing_efficiency == pytest.approx(0.52)
        assert lattice.neighbors_per_sphere == 6

    def test_create_bcc_lattice(self):
        """Test creating a BCC lattice."""
        lattice = SphereLattice(
            sphere_diameter=20.0,
            tube_diameter=4.0,
            lattice_type="bcc",
            extent=(2, 2, 2),
        )

        assert lattice.lattice_type == "bcc"
        assert lattice.packing_efficiency == pytest.approx(0.68)
        assert lattice.neighbors_per_sphere == 8

    def test_create_fcc_lattice(self):
        """Test creating a FCC lattice."""
        lattice = SphereLattice(
            sphere_diameter=20.0,
            tube_diameter=4.0,
            lattice_type="fcc",
            extent=(2, 2, 2),
        )

        assert lattice.lattice_type == "fcc"
        assert lattice.packing_efficiency == pytest.approx(0.74)
        assert lattice.neighbors_per_sphere == 12

    def test_auto_lattice_constant(self):
        """Test that lattice constant is auto-calculated correctly."""
        lattice = SphereLattice(
            sphere_diameter=20.0,
            tube_diameter=4.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
        )

        # Default: sphere_diameter + tube_diameter
        assert lattice.lattice_constant == 24.0
        assert lattice.tube_length == 4.0

    def test_explicit_lattice_constant(self):
        """Test setting explicit lattice constant."""
        lattice = SphereLattice(
            sphere_diameter=20.0,
            tube_diameter=4.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
            lattice_constant=30.0,
        )

        assert lattice.lattice_constant == 30.0
        assert lattice.tube_length == 10.0  # 30 - 20

    def test_sphere_positions_cubic(self):
        """Test sphere positions in cubic lattice."""
        lattice = SphereLattice(
            sphere_diameter=20.0,
            tube_diameter=4.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
        )

        positions = lattice.sphere_positions()

        # 2x2x2 cubic = 8 spheres
        assert len(positions) == 8

        # All positions should be within expected bounds
        a = lattice.lattice_constant
        for x, y, z in positions:
            assert 0 < x < 2 * a
            assert 0 < y < 2 * a
            assert 0 < z < 2 * a

    def test_sphere_positions_bcc(self):
        """Test sphere positions in BCC lattice."""
        lattice = SphereLattice(
            sphere_diameter=20.0,
            tube_diameter=4.0,
            lattice_type="bcc",
            extent=(2, 2, 2),
        )

        positions = lattice.sphere_positions()

        # BCC: 8 corners + some body centers
        # For 2x2x2, we get 8 corner positions + 1 body center = 9
        assert len(positions) == 9

    def test_sphere_positions_fcc(self):
        """Test sphere positions in FCC lattice."""
        lattice = SphereLattice(
            sphere_diameter=20.0,
            tube_diameter=4.0,
            lattice_type="fcc",
            extent=(2, 2, 2),
        )

        positions = lattice.sphere_positions()

        # FCC: corners + face centers
        # More spheres than cubic due to face centers
        assert len(positions) > 8

    def test_tube_connections_cubic(self):
        """Test tube connections in cubic lattice."""
        lattice = SphereLattice(
            sphere_diameter=20.0,
            tube_diameter=4.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
        )

        connections = lattice.tube_connections()

        # 2x2x2 cubic: each internal face has 4 tubes
        # 3 internal faces, 4 tubes each = 12 connections
        assert len(connections) == 12

    def test_resonant_frequency_calculation(self):
        """Test that resonant frequency matches analytical Helmholtz prediction."""
        lattice = SphereLattice(
            sphere_diameter=30.0,
            tube_diameter=6.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
        )

        f = lattice.resonant_frequency()

        # Calculate expected frequency manually
        c = SPEED_OF_SOUND
        V = (4 / 3) * np.pi * (15e-3) ** 3  # Sphere volume in m³
        S = np.pi * (3e-3) ** 2  # Tube area in m²
        L = lattice.tube_length / 1000  # Tube length in m
        r_tube = 3e-3
        L_eff = L + 2 * 0.85 * r_tube

        expected_f = (c / (2 * np.pi)) * np.sqrt(S / (V * L_eff))

        # Allow 5% tolerance as per acceptance criteria
        assert f == pytest.approx(expected_f, rel=0.05)

    def test_resonant_frequency_range(self):
        """Test that frequency scales appropriately with geometry."""
        # Larger sphere = lower frequency
        lattice_large = SphereLattice(
            sphere_diameter=40.0,
            tube_diameter=6.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
        )

        lattice_small = SphereLattice(
            sphere_diameter=20.0,
            tube_diameter=6.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
        )

        f_large = lattice_large.resonant_frequency()
        f_small = lattice_small.resonant_frequency()

        # Smaller cavity = higher frequency
        assert f_small > f_large

        # Both should be in audible range for these sizes
        assert 50 < f_large < 2000
        assert 50 < f_small < 2000

    def test_to_stack_generates_geometry(self):
        """Test that to_stack creates non-empty geometry."""
        lattice = SphereLattice(
            sphere_diameter=20.0,
            tube_diameter=4.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
        )

        stack = lattice.to_stack(resolution=1.0)

        # Should have multiple slices
        assert stack.num_slices >= 1

        # Should have some air regions (spheres + tubes)
        total_air = sum(s.mask.sum() for s in stack.slices)
        assert total_air > 0

    def test_to_stack_all_lattice_types(self):
        """Test that all lattice types generate valid geometry."""
        for lattice_type in ["cubic", "bcc", "fcc"]:
            lattice = SphereLattice(
                sphere_diameter=20.0,
                tube_diameter=4.0,
                lattice_type=lattice_type,
                extent=(2, 2, 2),
            )

            stack = lattice.to_stack(resolution=1.0)
            assert stack.num_slices >= 1

    def test_check_manufacturable_valid_lattice(self):
        """Test that valid lattice passes manufacturing checks."""
        lattice = SphereLattice(
            sphere_diameter=30.0,
            tube_diameter=4.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
        )

        violations = lattice.check_manufacturable(min_gap=2.0)
        assert len(violations) == 0

    def test_check_manufacturable_narrow_tubes(self):
        """Test that lattices with narrow tubes are flagged."""
        lattice = SphereLattice(
            sphere_diameter=30.0,
            tube_diameter=1.5,  # Below min_gap
            lattice_type="cubic",
            extent=(2, 2, 2),
        )

        violations = lattice.check_manufacturable(min_gap=2.0)
        assert any(v.constraint == "min_tube_diameter" for v in violations)

    def test_invalid_lattice_type_raises(self):
        """Test that invalid lattice type raises error."""
        with pytest.raises(ValueError, match="lattice_type"):
            SphereLattice(
                sphere_diameter=20.0,
                tube_diameter=4.0,
                lattice_type="invalid",
                extent=(2, 2, 2),
            )

    def test_invalid_dimensions_raise(self):
        """Test that invalid dimensions raise errors."""
        # Non-positive sphere diameter
        with pytest.raises(ValueError, match="sphere_diameter"):
            SphereLattice(
                sphere_diameter=0.0,
                tube_diameter=4.0,
                lattice_type="cubic",
                extent=(2, 2, 2),
            )

        # Non-positive tube diameter
        with pytest.raises(ValueError, match="tube_diameter"):
            SphereLattice(
                sphere_diameter=20.0,
                tube_diameter=-1.0,
                lattice_type="cubic",
                extent=(2, 2, 2),
            )

        # Invalid extent
        with pytest.raises(ValueError, match="extent"):
            SphereLattice(
                sphere_diameter=20.0,
                tube_diameter=4.0,
                lattice_type="cubic",
                extent=(0, 2, 2),
            )

    def test_lattice_constant_too_small_raises(self):
        """Test that lattice constant smaller than sphere diameter raises."""
        with pytest.raises(ValueError, match="lattice_constant"):
            SphereLattice(
                sphere_diameter=20.0,
                tube_diameter=4.0,
                lattice_type="cubic",
                extent=(2, 2, 2),
                lattice_constant=15.0,  # Smaller than sphere!
            )

    def test_frequency_scaling_with_geometry(self):
        """Test that frequency scales correctly with geometry parameters."""
        base_lattice = SphereLattice(
            sphere_diameter=30.0,
            tube_diameter=6.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
        )
        base_freq = base_lattice.resonant_frequency()

        # Double sphere volume -> frequency decreases by ~sqrt(2)
        larger_sphere = SphereLattice(
            sphere_diameter=30.0 * (2 ** (1 / 3)),  # Double volume
            tube_diameter=6.0,
            lattice_type="cubic",
            extent=(2, 2, 2),
        )
        larger_freq = larger_sphere.resonant_frequency()
        # f ∝ 1/sqrt(V), so doubling V halves f² -> f_new ≈ f_old/sqrt(2)
        assert larger_freq < base_freq

        # Double tube area -> frequency increases by ~sqrt(2)
        larger_tube = SphereLattice(
            sphere_diameter=30.0,
            tube_diameter=6.0 * np.sqrt(2),  # Double area
            lattice_type="cubic",
            extent=(2, 2, 2),
        )
        larger_tube_freq = larger_tube.resonant_frequency()
        assert larger_tube_freq > base_freq
