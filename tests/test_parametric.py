"""
Unit tests for parametric geometry.

Tests verify:
- Parameter validation and bounds checking
- ParametricHorn parameter management
- Constraint evaluation and feasibility checking
- GeometryOptimizer with simple objectives
"""

import importlib.util

import numpy as np
import pytest

from strata_fdtd import UniformGrid
from strata_fdtd.parametric import (
    ConstraintSet,
    GeometricConstraint,
    GeometryOptimizer,
    Parameter,
    ParametricHorn,
)

# =============================================================================
# Helper Functions
# =============================================================================


def _has_scipy():
    """Check if scipy is available."""
    return importlib.util.find_spec("scipy") is not None


# =============================================================================
# Parameter Tests
# =============================================================================


class TestParameter:
    def test_construction(self):
        """Test parameter construction with all fields."""
        param = Parameter(
            name="radius",
            default=0.02,
            bounds=(0.01, 0.05),
            unit="m",
            description="Throat radius",
        )

        assert param.name == "radius"
        assert param.default == 0.02
        assert param.bounds == (0.01, 0.05)
        assert param.unit == "m"
        assert param.description == "Throat radius"

    def test_validate_within_bounds(self):
        """Test validation accepts values within bounds."""
        param = Parameter("x", 1.0, (0.0, 2.0))

        assert param.validate(0.0) is True
        assert param.validate(1.0) is True
        assert param.validate(2.0) is True

    def test_validate_outside_bounds(self):
        """Test validation rejects values outside bounds."""
        param = Parameter("x", 1.0, (0.0, 2.0))

        assert param.validate(-0.1) is False
        assert param.validate(2.1) is False

    def test_clip(self):
        """Test clipping values to bounds."""
        param = Parameter("x", 1.0, (0.0, 2.0))

        assert param.clip(-0.5) == 0.0
        assert param.clip(1.0) == 1.0
        assert param.clip(3.0) == 2.0


# =============================================================================
# ParametricHorn Tests
# =============================================================================


class TestParametricHorn:
    def test_construction_defaults(self):
        """Test construction with default values."""
        horn = ParametricHorn()

        assert horn.throat_radius == 0.02
        assert horn.mouth_radius == 0.06
        assert horn.length == 0.15
        assert horn.profile == "exponential"
        np.testing.assert_allclose(horn.axis_start, [0, 0, 0])
        np.testing.assert_allclose(horn.axis_direction, [0, 0, 1])

    def test_construction_custom(self):
        """Test construction with custom values."""
        horn = ParametricHorn(
            throat_radius=0.03,
            mouth_radius=0.08,
            length=0.2,
            profile="conical",
        )

        assert horn.throat_radius == 0.03
        assert horn.mouth_radius == 0.08
        assert horn.length == 0.2
        assert horn.profile == "conical"

    def test_axis_normalization(self):
        """Test that axis direction is normalized on construction."""
        horn = ParametricHorn(axis_direction=np.array([1, 1, 1]))

        norm = np.linalg.norm(horn.axis_direction)
        assert abs(norm - 1.0) < 1e-10

    def test_get_parameters(self):
        """Test parameter retrieval."""
        horn = ParametricHorn(throat_radius=0.02, mouth_radius=0.06, length=0.15)
        params = horn.get_parameters()

        assert len(params) == 3
        assert params[0].name == "throat_radius"
        assert params[0].default == 0.02
        assert params[1].name == "mouth_radius"
        assert params[1].default == 0.06
        assert params[2].name == "length"
        assert params[2].default == 0.15

    def test_to_dict(self):
        """Test parameter export to dict."""
        horn = ParametricHorn(throat_radius=0.03, mouth_radius=0.07, length=0.2)
        params_dict = horn.to_dict()

        assert params_dict["throat_radius"] == 0.03
        assert params_dict["mouth_radius"] == 0.07
        assert params_dict["length"] == 0.2

    def test_with_parameters(self):
        """Test creating new instance with updated parameters."""
        horn1 = ParametricHorn(throat_radius=0.02, mouth_radius=0.06, length=0.15)
        horn2 = horn1.with_parameters(mouth_radius=0.08, length=0.2)

        # Original unchanged
        assert horn1.mouth_radius == 0.06
        assert horn1.length == 0.15

        # New instance has updated values
        assert horn2.throat_radius == 0.02  # Unchanged
        assert horn2.mouth_radius == 0.08  # Updated
        assert horn2.length == 0.2  # Updated

    def test_to_sdf(self):
        """Test conversion to concrete SDF primitive."""
        horn_param = ParametricHorn(
            throat_radius=0.025,
            mouth_radius=0.075,
            length=0.15,
            profile="exponential",
        )

        horn_sdf = horn_param.to_sdf()

        # Check that it's a Horn primitive
        from strata_fdtd.sdf import Horn

        assert isinstance(horn_sdf, Horn)
        assert horn_sdf.throat_radius == 0.025
        assert horn_sdf.mouth_radius == 0.075
        assert horn_sdf.profile == "exponential"

    def test_voxelize(self):
        """Test voxelization through parametric interface."""
        horn = ParametricHorn(throat_radius=0.02, mouth_radius=0.06, length=0.1)
        grid = UniformGrid(shape=(50, 50, 50), resolution=2e-3)

        mask = horn.voxelize(grid)

        assert mask.shape == (50, 50, 50)
        assert mask.dtype == bool
        # Should have some True values (air inside horn)
        assert mask.sum() > 0


# =============================================================================
# GeometricConstraint Tests
# =============================================================================


class TestGeometricConstraint:
    def test_construction(self):
        """Test constraint construction."""

        def expr(params):
            return params["x"] - 10

        constraint = GeometricConstraint(
            name="max_x",
            expression=expr,
            description="x must be <= 10",
        )

        assert constraint.name == "max_x"
        assert constraint.description == "x must be <= 10"

    def test_evaluate_satisfied(self):
        """Test evaluation when constraint is satisfied."""

        def expr(params):
            return params["x"] - 10  # Violation if x > 10

        constraint = GeometricConstraint("max_x", expr)

        # x = 5: constraint satisfied (violation = 0)
        assert constraint.evaluate({"x": 5}) == 0

    def test_evaluate_violated(self):
        """Test evaluation when constraint is violated."""

        def expr(params):
            return params["x"] - 10  # Violation if x > 10

        constraint = GeometricConstraint("max_x", expr)

        # x = 15: constraint violated (violation = 5)
        violation = constraint.evaluate({"x": 15})
        assert abs(violation - 5) < 1e-10


# =============================================================================
# ConstraintSet Tests
# =============================================================================


class TestConstraintSet:
    def test_construction(self):
        """Test constraint set construction."""
        cs = ConstraintSet()
        assert len(cs.constraints) == 0

    def test_add_constraint(self):
        """Test adding constraints."""
        cs = ConstraintSet()
        cs.add("test", lambda p: p["x"] - 5, "x <= 5")

        assert len(cs.constraints) == 1
        assert cs.constraints[0].name == "test"

    def test_add_min_value(self):
        """Test adding minimum value constraint."""
        cs = ConstraintSet()
        cs.add_min_value("length", 0.1)

        # length = 0.15: satisfied
        assert cs.evaluate_all({"length": 0.15})["min_length"] == 0

        # length = 0.05: violated by 0.05
        violation = cs.evaluate_all({"length": 0.05})["min_length"]
        assert abs(violation - 0.05) < 1e-10

    def test_add_max_value(self):
        """Test adding maximum value constraint."""
        cs = ConstraintSet()
        cs.add_max_value("radius", 0.1)

        # radius = 0.08: satisfied
        assert cs.evaluate_all({"radius": 0.08})["max_radius"] == 0

        # radius = 0.15: violated by 0.05
        violation = cs.evaluate_all({"radius": 0.15})["max_radius"]
        assert abs(violation - 0.05) < 1e-10

    def test_add_ratio_constraint(self):
        """Test adding ratio constraint."""
        cs = ConstraintSet()
        # mouth_radius >= throat_radius * 2
        cs.add_ratio_constraint("throat_radius", "mouth_radius", 2.0)

        # throat=0.02, mouth=0.05: satisfied (ratio=2.5)
        violations = cs.evaluate_all({"throat_radius": 0.02, "mouth_radius": 0.05})
        assert violations["mouth_radius_over_throat_radius"] == 0

        # throat=0.02, mouth=0.03: violated (ratio=1.5)
        violations = cs.evaluate_all({"throat_radius": 0.02, "mouth_radius": 0.03})
        assert violations["mouth_radius_over_throat_radius"] > 0

    def test_is_feasible(self):
        """Test feasibility checking."""
        cs = ConstraintSet()
        cs.add_min_value("x", 0.0)
        cs.add_max_value("x", 1.0)

        assert cs.is_feasible({"x": 0.5}) is True
        assert cs.is_feasible({"x": -0.1}) is False
        assert cs.is_feasible({"x": 1.5}) is False

    def test_total_violation(self):
        """Test total violation computation."""
        cs = ConstraintSet()
        cs.add("c1", lambda p: p["x"] - 5)  # Violated by 5 if x=10
        cs.add("c2", lambda p: p["y"] - 3)  # Violated by 2 if y=5

        total = cs.total_violation({"x": 10, "y": 5})
        assert abs(total - 7) < 1e-10


# =============================================================================
# GeometryOptimizer Tests
# =============================================================================


class TestGeometryOptimizer:
    def test_construction(self):
        """Test optimizer construction."""
        horn = ParametricHorn()
        grid = UniformGrid(shape=(50, 50, 50), resolution=2e-3)
        optimizer = GeometryOptimizer(horn, grid)

        assert optimizer.geometry is horn
        assert optimizer.grid is grid
        assert len(optimizer._param_names) == 3

    def test_objective_evaluation(self):
        """Test objective function evaluation."""
        horn = ParametricHorn(throat_radius=0.02, mouth_radius=0.06, length=0.1)
        grid = UniformGrid(shape=(50, 50, 50), resolution=2e-3)
        optimizer = GeometryOptimizer(horn, grid)

        # Simple objective: negative volume (maximize volume)
        def objective(mask):
            return -mask.sum()

        # Evaluate at initial parameters
        x0 = np.array([0.02, 0.06, 0.1])
        obj_value = optimizer.objective(x0, objective)

        # Should be negative (valid geometry)
        assert obj_value < 0

    def test_objective_with_constraints(self):
        """Test objective evaluation with constraints."""
        horn = ParametricHorn()
        grid = UniformGrid(shape=(50, 50, 50), resolution=2e-3)

        # Add constraint: mouth must be at least 2x throat
        constraints = ConstraintSet()
        constraints.add_ratio_constraint("throat_radius", "mouth_radius", 2.0)

        optimizer = GeometryOptimizer(horn, grid, constraints)

        def objective(mask):
            return -mask.sum()

        # Feasible: throat=0.02, mouth=0.06, length=0.1
        x_feasible = np.array([0.02, 0.06, 0.1])
        obj_feasible = optimizer.objective(x_feasible, objective)
        assert obj_feasible < 1e5  # Not a penalty

        # Infeasible: throat=0.05, mouth=0.06 (ratio < 2)
        x_infeasible = np.array([0.05, 0.06, 0.1])
        obj_infeasible = optimizer.objective(x_infeasible, objective)
        assert obj_infeasible > 1e5  # Large penalty

    @pytest.mark.skipif(
        not _has_scipy(),
        reason="scipy required for optimization",
    )
    def test_optimize_simple(self):
        """Test optimization with simple objective."""
        horn = ParametricHorn(throat_radius=0.02, mouth_radius=0.04, length=0.1)
        grid = UniformGrid(shape=(30, 30, 40), resolution=2e-3)
        optimizer = GeometryOptimizer(horn, grid)

        # Objective: maximize mouth radius (within bounds)
        def objective(mask):
            # Use negative sum as proxy for size
            return -mask.sum()

        result = optimizer.optimize(
            objective,
            method="Nelder-Mead",
            options={"maxiter": 50, "xatol": 1e-3},
        )

        # Check result structure
        assert "parameters" in result
        assert "objective" in result
        assert "success" in result
        assert "geometry" in result

        # Optimal mouth should be larger than initial
        assert result["parameters"]["mouth_radius"] >= 0.04
