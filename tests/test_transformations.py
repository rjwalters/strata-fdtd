"""
Unit tests for SDF transformations (Translate, Scale, Rotate).

Tests verify:
- Transformation correctness for known transforms
- Bounding box propagation
- Fluent API methods
- Transform composition
- Edge cases and parameter validation
"""

import numpy as np
import pytest

from strata_fdtd import Box, Cylinder, Sphere
from strata_fdtd.geometry.sdf import Rotate, Scale, Translate, rotation_matrix

# =============================================================================
# rotation_matrix Tests
# =============================================================================


class TestRotationMatrix:
    def test_rotation_around_z_axis(self):
        """Test rotation matrix for 90° around z-axis."""
        R = rotation_matrix(np.array([0, 0, 1]), np.pi / 2)

        # Rotate point (1, 0, 0) → (0, 1, 0)
        point = np.array([1, 0, 0])
        rotated = R @ point

        np.testing.assert_allclose(rotated, [0, 1, 0], atol=1e-10)

    def test_rotation_around_x_axis(self):
        """Test rotation matrix for 90° around x-axis."""
        R = rotation_matrix(np.array([1, 0, 0]), np.pi / 2)

        # Rotate point (0, 1, 0) → (0, 0, 1)
        point = np.array([0, 1, 0])
        rotated = R @ point

        np.testing.assert_allclose(rotated, [0, 0, 1], atol=1e-10)

    def test_rotation_around_arbitrary_axis(self):
        """Test rotation around arbitrary axis."""
        # Rotate around (1, 1, 1) axis
        axis = np.array([1, 1, 1])
        R = rotation_matrix(axis, np.pi / 3)

        # Check that rotation matrix is orthogonal (R @ R.T = I)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)

        # Check determinant is 1 (proper rotation)
        assert abs(np.linalg.det(R) - 1.0) < 1e-10

    def test_rotation_normalizes_axis(self):
        """Test that axis is automatically normalized."""
        # Non-unit axis
        R1 = rotation_matrix(np.array([2, 0, 0]), np.pi / 2)
        # Unit axis
        R2 = rotation_matrix(np.array([1, 0, 0]), np.pi / 2)

        # Should be identical
        np.testing.assert_allclose(R1, R2, atol=1e-10)

    def test_zero_rotation(self):
        """Test that zero angle gives identity matrix."""
        R = rotation_matrix(np.array([1, 0, 0]), 0.0)

        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


# =============================================================================
# Translate Tests
# =============================================================================


class TestTranslate:
    def test_translate_box(self):
        """Test translating a box."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))
        offset = (0.05, 0.05, 0.05)
        translated = Translate(box, offset)

        # Point that was inside original box
        points = np.array([[0.0, 0.0, 0.0]])
        dist_original = box.sdf(points)

        # Same point relative to translated box should be outside
        dist_translated = translated.sdf(points)

        # Original center is inside
        assert dist_original < 0
        # After translation, original center is outside
        assert dist_translated > 0

    def test_translate_sdf_consistency(self):
        """Test that translation preserves SDF distances correctly."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))
        offset = np.array([0.05, 0.05, 0.05])
        translated = Translate(box, offset)

        # Test points
        points = np.array([[0.05, 0.05, 0.05]])  # Center of translated box

        # Distance at translated center should equal distance at original center
        dist_at_translated_center = translated.sdf(points)
        dist_at_original_center = box.sdf(np.array([[0.0, 0.0, 0.0]]))

        np.testing.assert_allclose(dist_at_translated_center, dist_at_original_center, atol=1e-10)

    def test_translate_bounding_box(self):
        """Test that bounding box is correctly translated."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))
        offset = np.array([0.05, 0.05, 0.05])
        translated = Translate(box, offset)

        bb_min, bb_max = translated.bounding_box

        expected_min = np.array([-0.01, -0.01, -0.01]) + offset
        expected_max = np.array([0.01, 0.01, 0.01]) + offset

        np.testing.assert_allclose(bb_min, expected_min, atol=1e-10)
        np.testing.assert_allclose(bb_max, expected_max, atol=1e-10)

    def test_translate_fluent_api(self):
        """Test fluent API for translation."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))
        offset = (0.05, 0.05, 0.05)

        # Using fluent API
        translated = box.translate(offset)

        # Should behave identically to Translate(box, offset)
        direct = Translate(box, offset)

        points = np.array([[0.05, 0.05, 0.05]])
        np.testing.assert_allclose(
            translated.sdf(points), direct.sdf(points), atol=1e-10
        )

    def test_translate_rejects_invalid_offset(self):
        """Test that invalid offset shapes are rejected."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))

        with pytest.raises(ValueError, match="must be a 3-element vector"):
            Translate(box, [1, 2])  # Only 2 elements

        with pytest.raises(ValueError, match="must be a 3-element vector"):
            Translate(box, [1, 2, 3, 4])  # Too many elements


# =============================================================================
# Scale Tests
# =============================================================================


class TestScale:
    def test_scale_uniform(self):
        """Test uniform scaling of a box."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))
        scaled = Scale(box, scale=2.0)

        # Points on original box surface
        points = np.array([[0.01, 0.0, 0.0]])  # Surface at x=0.01

        # For uniform scale of 2x, surface should now be at x=0.02
        scaled_surface = np.array([[0.02, 0.0, 0.0]])

        dist_original_surface = box.sdf(points)
        dist_scaled_surface = scaled.sdf(scaled_surface)

        # Both should be on surface (distance ~0)
        assert abs(dist_original_surface) < 1e-10
        assert abs(dist_scaled_surface) < 1e-10

    def test_scale_preserves_sdf_for_uniform(self):
        """Test that uniform scaling preserves SDF property."""
        sphere = Sphere(center=(0, 0, 0), radius=0.01)
        scale_factor = 3.0
        scaled = Scale(sphere, scale=scale_factor)

        # Point outside original sphere
        points = np.array([[0.02, 0.0, 0.0]])

        # Distance should scale by the same factor
        dist_original = sphere.sdf(points)
        dist_scaled = scaled.sdf(points * scale_factor)

        np.testing.assert_allclose(
            dist_scaled, dist_original * scale_factor, atol=1e-10
        )

    def test_scale_non_uniform(self):
        """Test non-uniform scaling."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))
        scaled = Scale(box, scale=(2.0, 1.0, 0.5))

        # Bounding box should reflect non-uniform scaling
        bb_min, bb_max = scaled.bounding_box

        expected_min = np.array([-0.01, -0.01, -0.01]) * np.array([2.0, 1.0, 0.5])
        expected_max = np.array([0.01, 0.01, 0.01]) * np.array([2.0, 1.0, 0.5])

        np.testing.assert_allclose(bb_min, expected_min, atol=1e-10)
        np.testing.assert_allclose(bb_max, expected_max, atol=1e-10)

    def test_scale_bounding_box(self):
        """Test that bounding box is correctly scaled."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))
        scale_factor = 2.5
        scaled = Scale(box, scale=scale_factor)

        bb_min, bb_max = scaled.bounding_box

        original_min, original_max = box.bounding_box
        expected_min = original_min * scale_factor
        expected_max = original_max * scale_factor

        np.testing.assert_allclose(bb_min, expected_min, atol=1e-10)
        np.testing.assert_allclose(bb_max, expected_max, atol=1e-10)

    def test_scale_fluent_api(self):
        """Test fluent API for scaling."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))

        # Using fluent API
        scaled = box.scale(2.0)

        # Should behave identically to Scale(box, 2.0)
        direct = Scale(box, 2.0)

        points = np.array([[0.0, 0.0, 0.0]])
        np.testing.assert_allclose(scaled.sdf(points), direct.sdf(points), atol=1e-10)

    def test_scale_rejects_negative(self):
        """Test that negative scale factors are rejected."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))

        with pytest.raises(ValueError, match="must be positive"):
            Scale(box, scale=-1.0)

        with pytest.raises(ValueError, match="must be positive"):
            Scale(box, scale=(1.0, -1.0, 1.0))

    def test_scale_rejects_zero(self):
        """Test that zero scale factors are rejected."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))

        with pytest.raises(ValueError, match="must be positive"):
            Scale(box, scale=0.0)


# =============================================================================
# Rotate Tests
# =============================================================================


class TestRotate:
    def test_rotate_box_90_degrees(self):
        """Test rotating a box 90° around z-axis."""
        # Elongated box along x-axis
        box = Box(center=(0, 0, 0), size=(0.04, 0.02, 0.02))

        # Rotate 90° around z-axis
        rotated = Rotate(box, axis=(0, 0, 1), angle=np.pi / 2)

        # Point that was on +x surface should now be on +y surface
        # Original: surface at x=0.02
        # After rotation: should be at y=0.02

        points = np.array([[0.0, 0.02, 0.0]])
        dist = rotated.sdf(points)

        # Should be on surface (distance ~0)
        assert abs(dist) < 1e-10

    def test_rotate_with_matrix(self):
        """Test rotation using explicit matrix."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))

        # Create rotation matrix for 90° around z
        R = rotation_matrix(np.array([0, 0, 1]), np.pi / 2)

        # Create rotation using matrix
        rotated = Rotate(box, matrix=R)

        # Should match axis-angle rotation
        rotated_axis = Rotate(box, axis=(0, 0, 1), angle=np.pi / 2)

        points = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [-0.01, 0.0, 0.0]])
        np.testing.assert_allclose(
            rotated.sdf(points), rotated_axis.sdf(points), atol=1e-10
        )

    def test_rotate_bounding_box(self):
        """Test that bounding box is correct after rotation."""
        # Create elongated box along x-axis
        box = Box(center=(0, 0, 0), size=(0.04, 0.02, 0.02))

        # Rotate 90° around z-axis (x → y)
        rotated = Rotate(box, axis=(0, 0, 1), angle=np.pi / 2)

        bb_min, bb_max = rotated.bounding_box

        # After rotation, box should be elongated along y-axis
        # Expected: x in [-0.01, 0.01], y in [-0.02, 0.02], z in [-0.01, 0.01]
        expected_min = np.array([-0.01, -0.02, -0.01])
        expected_max = np.array([0.01, 0.02, 0.01])

        np.testing.assert_allclose(bb_min, expected_min, atol=1e-10)
        np.testing.assert_allclose(bb_max, expected_max, atol=1e-10)

    def test_rotate_fluent_api(self):
        """Test fluent API for rotation."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))

        # Using fluent API
        rotated = box.rotate((0, 0, 1), np.pi / 2)

        # Should behave identically to Rotate(box, ...)
        direct = Rotate(box, axis=(0, 0, 1), angle=np.pi / 2)

        points = np.array([[0.01, 0.0, 0.0]])
        np.testing.assert_allclose(rotated.sdf(points), direct.sdf(points), atol=1e-10)

    def test_rotate_x_convenience(self):
        """Test rotate_x convenience method."""
        box = Box(center=(0, 0, 0), size=(0.04, 0.02, 0.02))

        # Using convenience method
        rotated = box.rotate_x(np.pi / 2)

        # Should be equivalent to rotate((1, 0, 0), angle)
        direct = box.rotate((1, 0, 0), np.pi / 2)

        points = np.array([[0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
        np.testing.assert_allclose(rotated.sdf(points), direct.sdf(points), atol=1e-10)

    def test_rotate_y_convenience(self):
        """Test rotate_y convenience method."""
        box = Box(center=(0, 0, 0), size=(0.04, 0.02, 0.02))

        rotated = box.rotate_y(np.pi / 2)
        direct = box.rotate((0, 1, 0), np.pi / 2)

        points = np.array([[0.01, 0.0, 0.0]])
        np.testing.assert_allclose(rotated.sdf(points), direct.sdf(points), atol=1e-10)

    def test_rotate_z_convenience(self):
        """Test rotate_z convenience method."""
        box = Box(center=(0, 0, 0), size=(0.04, 0.02, 0.02))

        rotated = box.rotate_z(np.pi / 2)
        direct = box.rotate((0, 0, 1), np.pi / 2)

        points = np.array([[0.01, 0.0, 0.0]])
        np.testing.assert_allclose(rotated.sdf(points), direct.sdf(points), atol=1e-10)

    def test_rotate_requires_parameters(self):
        """Test that rotation requires either axis/angle or matrix."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))

        with pytest.raises(ValueError, match="Must provide either"):
            Rotate(box)  # No parameters

        with pytest.raises(ValueError, match="Must provide either"):
            Rotate(box, axis=(0, 0, 1))  # Missing angle

        with pytest.raises(ValueError, match="Must provide either"):
            Rotate(box, angle=np.pi / 2)  # Missing axis

    def test_rotate_rejects_invalid_matrix(self):
        """Test that invalid matrix shapes are rejected."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))

        with pytest.raises(ValueError, match="must be 3x3"):
            Rotate(box, matrix=np.eye(2))  # 2x2 matrix

        with pytest.raises(ValueError, match="must be 3x3"):
            Rotate(box, matrix=np.eye(4))  # 4x4 matrix


# =============================================================================
# Transform Composition Tests
# =============================================================================


class TestTransformComposition:
    def test_translate_then_rotate(self):
        """Test composing translation followed by rotation."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))

        # Translate then rotate
        translated = box.translate((0.05, 0.0, 0.0))
        rotated = translated.rotate_z(np.pi / 2)

        # The box center should now be at (0, 0.05, 0)
        points = np.array([[0.0, 0.05, 0.0]])
        dist = rotated.sdf(points)

        # Center should be inside
        assert dist < 0

    def test_rotate_then_translate(self):
        """Test composing rotation followed by translation."""
        # Elongated box along x
        box = Box(center=(0, 0, 0), size=(0.04, 0.02, 0.02))

        # Rotate then translate
        rotated = box.rotate_z(np.pi / 2)  # Now along y
        translated = rotated.translate((0.05, 0.05, 0.0))

        # Box center at (0.05, 0.05, 0), elongated along y
        # Point at (0.05, 0.07, 0) should be on surface
        points = np.array([[0.05, 0.07, 0.0]])
        dist = translated.sdf(points)

        assert abs(dist) < 1e-10

    def test_scale_then_translate(self):
        """Test composing scale followed by translation."""
        box = Box(center=(0, 0, 0), size=(0.02, 0.02, 0.02))

        # Scale then translate
        scaled = box.scale(2.0)
        translated = scaled.translate((0.1, 0.0, 0.0))

        # Scaled box has size 0.04, translated to center (0.1, 0, 0)
        # Surface should be at x = 0.1 ± 0.02
        points = np.array([[0.12, 0.0, 0.0]])
        dist = translated.sdf(points)

        assert abs(dist) < 1e-10

    def test_complex_composition(self):
        """Test complex transformation chain."""
        cylinder = Cylinder(p1=(0, 0, 0), p2=(0, 0, 0.1), radius=0.01)

        # Complex chain: rotate, scale, translate
        transformed = (
            cylinder.rotate_x(np.radians(15))  # Tilt 15 degrees
            .scale(1.5)  # Scale up 1.5x
            .translate((0.05, 0.05, 0.02))  # Move to position
        )

        # Should still be a valid SDF (bounding box should be computable)
        bb_min, bb_max = transformed.bounding_box

        # Bounding box should be reasonable (not NaN, not too large)
        assert not np.any(np.isnan(bb_min))
        assert not np.any(np.isnan(bb_max))
        assert np.all(bb_min < bb_max)


# =============================================================================
# Integration Tests
# =============================================================================


class TestTransformIntegration:
    def test_example_from_issue(self):
        """Test the example from issue #168."""
        # Create a port tube angled 15 degrees
        port = Cylinder(p1=(0, 0, 0), p2=(0, 0, 0.1), radius=0.02)
        angled_port = port.rotate_x(np.radians(15)).translate((0.05, 0.05, 0.02))

        # Should have valid bounding box
        bb_min, bb_max = angled_port.bounding_box
        assert np.all(bb_min < bb_max)

        # Should be able to evaluate SDF
        points = np.array([[0.05, 0.05, 0.02]])
        dist = angled_port.sdf(points)
        assert not np.isnan(dist[0])

    def test_mirroring_with_negative_scale(self):
        """Test mirroring geometry with negative scale (if supported)."""
        # Note: Current implementation rejects negative scale
        # This test documents that behavior
        box = Box(center=(0.05, 0.05, 0.05), size=(0.02, 0.02, 0.02))

        with pytest.raises(ValueError, match="must be positive"):
            box.scale((-1, 1, 1))

    def test_transformed_contains(self):
        """Test that contains() works on transformed primitives."""
        sphere = Sphere(center=(0, 0, 0), radius=0.01)
        translated = sphere.translate((0.05, 0.05, 0.05))

        points = np.array([
            [0.05, 0.05, 0.05],  # Inside (new center)
            [0.0, 0.0, 0.0],  # Outside (old center)
        ])

        inside = translated.contains(points)

        assert inside[0]  # New center is inside
        assert not inside[1]  # Old center is outside
