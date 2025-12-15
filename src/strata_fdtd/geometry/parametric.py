"""
Parametric geometry for optimization.

This module provides parametric variants of geometric primitives that enable
optimization over geometric dimensions. Parameters can be tuned individually
or through optimization algorithms to find optimal designs.

Classes:
    Parameter: Named geometric parameter with bounds
    ParametricPrimitive: Base class for parametric primitives
    ParametricHorn: Horn with tunable dimensions
    GeometricConstraint: Constraint on geometric parameters
    ConstraintSet: Collection of constraints
    GeometryOptimizer: Wrapper for scipy.optimize

Example:
    >>> from strata_fdtd.parametric import ParametricHorn, GeometryOptimizer
    >>> from strata_fdtd import UniformGrid
    >>>
    >>> # Define parametric horn
    >>> horn = ParametricHorn(throat_radius=0.02, mouth_radius=0.06, length=0.15)
    >>>
    >>> # Create optimizer
    >>> grid = UniformGrid(shape=(100, 100, 150), resolution=2e-3)
    >>> optimizer = GeometryOptimizer(horn, grid)
    >>>
    >>> # Optimize for some objective
    >>> result = optimizer.optimize(lambda mask: -mask.sum())
    >>> print(f"Optimal: {result['parameters']}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from strata_fdtd.grid import UniformGrid
    from strata_fdtd.geometry.sdf import Horn, SDFPrimitive


@dataclass
class Parameter:
    """A named geometric parameter with bounds and constraints.

    Parameters represent tunable dimensions in geometric primitives,
    with valid ranges and optional units and descriptions.

    Args:
        name: Parameter name (e.g., "throat_radius")
        default: Default value
        bounds: (min, max) valid range
        unit: Physical unit (e.g., "m", "Hz", "")
        description: Human-readable description

    Example:
        >>> param = Parameter(
        ...     name="throat_radius",
        ...     default=0.02,
        ...     bounds=(0.005, 0.05),
        ...     unit="m",
        ...     description="Horn throat radius",
        ... )
        >>> param.validate(0.03)  # True
        >>> param.validate(0.1)   # False (out of bounds)
    """

    name: str
    default: float
    bounds: tuple[float, float]
    unit: str = "m"
    description: str = ""

    def validate(self, value: float) -> bool:
        """Check if value is within bounds.

        Args:
            value: Value to check

        Returns:
            True if value is within [min, max] bounds
        """
        return self.bounds[0] <= value <= self.bounds[1]

    def clip(self, value: float) -> float:
        """Clip value to bounds.

        Args:
            value: Value to clip

        Returns:
            Value clipped to [min, max] range
        """
        return np.clip(value, self.bounds[0], self.bounds[1])


class ParametricPrimitive(ABC):
    """Base class for primitives with named tunable parameters.

    Parametric primitives can be converted to concrete SDF primitives
    and support parameter introspection for optimization.

    All subclasses must implement:
        - get_parameters(): List of tunable parameters
        - with_parameters(**kwargs): Create new instance with updated params
        - to_sdf(): Convert to concrete SDFPrimitive

    Example:
        >>> class ParametricBox(ParametricPrimitive):
        ...     def __init__(self, width=1.0, height=1.0):
        ...         self.width = width
        ...         self.height = height
        ...
        ...     def get_parameters(self):
        ...         return [
        ...             Parameter("width", self.width, (0.1, 2.0)),
        ...             Parameter("height", self.height, (0.1, 2.0)),
        ...         ]
        ...
        ...     def with_parameters(self, **kwargs):
        ...         params = self.to_dict()
        ...         params.update(kwargs)
        ...         return ParametricBox(**params)
        ...
        ...     def to_sdf(self):
        ...         return Box(center=(0, 0, 0), size=(self.width, self.height, 1.0))
    """

    @abstractmethod
    def get_parameters(self) -> list[Parameter]:
        """Return list of tunable parameters.

        Returns:
            List of Parameter objects defining tunable dimensions
        """
        pass

    @abstractmethod
    def with_parameters(self, **kwargs) -> ParametricPrimitive:
        """Return new instance with updated parameters.

        Args:
            **kwargs: Parameter name-value pairs to update

        Returns:
            New instance with updated parameters
        """
        pass

    @abstractmethod
    def to_sdf(self) -> SDFPrimitive:
        """Convert to concrete SDFPrimitive for voxelization.

        Returns:
            Concrete SDF primitive with current parameter values
        """
        pass

    def to_dict(self) -> dict:
        """Export current parameter values.

        Returns:
            Dict mapping parameter names to values
        """
        return {p.name: getattr(self, p.name) for p in self.get_parameters()}

    @classmethod
    def from_dict(cls, params: dict) -> ParametricPrimitive:
        """Create instance from parameter dict.

        Args:
            params: Dict mapping parameter names to values

        Returns:
            New instance with specified parameter values
        """
        return cls(**params)

    def voxelize(self, grid: UniformGrid) -> NDArray[np.bool_]:
        """Voxelize to boolean mask using current parameters.

        Args:
            grid: Grid defining voxel structure

        Returns:
            Boolean array where True = inside (air)
        """
        return self.to_sdf().voxelize(grid)


@dataclass
class ParametricHorn(ParametricPrimitive):
    """Horn with parametric dimensions.

    A horn primitive where throat/mouth radii and length can be tuned
    for optimization. The profile type and axis orientation are fixed.

    Args:
        throat_radius: Radius at throat (narrow end)
        mouth_radius: Radius at mouth (wide end)
        length: Horn length along axis
        profile: Flare profile ("exponential", "conical", "hyperbolic", "tractrix")
        axis_start: Start point of horn axis
        axis_direction: Direction of horn axis (will be normalized)

    Example:
        >>> horn = ParametricHorn(
        ...     throat_radius=0.02,
        ...     mouth_radius=0.06,
        ...     length=0.15,
        ...     profile="exponential",
        ... )
        >>> # Get tunable parameters
        >>> params = horn.get_parameters()
        >>> # Create variant with larger mouth
        >>> horn2 = horn.with_parameters(mouth_radius=0.08)
        >>> # Convert to concrete SDF
        >>> sdf = horn.to_sdf()
    """

    throat_radius: float = 0.02
    mouth_radius: float = 0.06
    length: float = 0.15
    profile: str = "exponential"

    # Fixed properties (not tunable)
    axis_start: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    axis_direction: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    def __post_init__(self):
        """Normalize axis direction after initialization."""
        self.axis_start = np.asarray(self.axis_start, dtype=np.float64)
        self.axis_direction = np.asarray(self.axis_direction, dtype=np.float64)
        norm = np.linalg.norm(self.axis_direction)
        if norm > 0:
            self.axis_direction = self.axis_direction / norm

    def get_parameters(self) -> list[Parameter]:
        """Return tunable parameters for optimization.

        Returns:
            List of throat_radius, mouth_radius, length parameters
        """
        return [
            Parameter(
                "throat_radius",
                self.throat_radius,
                (0.005, 0.05),
                "m",
                "Throat radius",
            ),
            Parameter(
                "mouth_radius",
                self.mouth_radius,
                (0.02, 0.15),
                "m",
                "Mouth radius",
            ),
            Parameter(
                "length",
                self.length,
                (0.05, 0.5),
                "m",
                "Horn length",
            ),
        ]

    def with_parameters(self, **kwargs) -> ParametricHorn:
        """Create new instance with updated parameters.

        Args:
            **kwargs: Parameter updates (throat_radius, mouth_radius, length)

        Returns:
            New ParametricHorn with updated values
        """
        # Start with current values
        params = {
            "throat_radius": self.throat_radius,
            "mouth_radius": self.mouth_radius,
            "length": self.length,
            "profile": self.profile,
            "axis_start": self.axis_start.copy(),
            "axis_direction": self.axis_direction.copy(),
        }
        # Update with provided kwargs
        params.update(kwargs)
        return ParametricHorn(**params)

    def to_sdf(self) -> Horn:
        """Convert to concrete Horn primitive.

        Returns:
            Horn with current parameter values
        """
        from strata_fdtd.geometry.sdf import Horn

        throat_position = self.axis_start
        mouth_position = self.axis_start + self.axis_direction * self.length

        return Horn(
            throat_position=tuple(throat_position),
            mouth_position=tuple(mouth_position),
            throat_radius=self.throat_radius,
            mouth_radius=self.mouth_radius,
            profile=self.profile,
        )


@dataclass
class GeometricConstraint:
    """Constraint on geometric parameters.

    Constraints enforce manufacturing limits, physical requirements,
    or design rules on parameter combinations.

    Args:
        name: Constraint name
        expression: Function mapping params dict to violation amount
                    (returns 0 if satisfied, >0 if violated)
        description: Human-readable description

    Example:
        >>> def mouth_larger_than_throat(params):
        ...     # Violated if throat >= mouth
        ...     return max(0, params["throat_radius"] - params["mouth_radius"] + 0.01)
        >>>
        >>> constraint = GeometricConstraint(
        ...     name="mouth_larger",
        ...     expression=mouth_larger_than_throat,
        ...     description="Mouth must be 1cm larger than throat",
        ... )
        >>> constraint.evaluate({"throat_radius": 0.02, "mouth_radius": 0.06})  # 0.0
    """

    name: str
    expression: Callable[[dict], float]
    description: str = ""

    def evaluate(self, params: dict) -> float:
        """Evaluate constraint violation.

        Args:
            params: Dict of parameter name-value pairs

        Returns:
            0 if constraint satisfied, >0 if violated
        """
        return max(0, self.expression(params))


class ConstraintSet:
    """Collection of geometric constraints.

    Manages multiple constraints and provides convenience methods
    for common manufacturing and design constraints.

    Example:
        >>> constraints = ConstraintSet()
        >>> # Add custom constraint
        >>> constraints.add(
        ...     "min_length",
        ...     lambda p: 0.05 - p.get("length", 0),
        ...     "Length must be at least 5cm",
        ... )
        >>> # Check all constraints
        >>> params = {"length": 0.15}
        >>> violations = constraints.evaluate_all(params)
        >>> feasible = constraints.is_feasible(params)
    """

    def __init__(self):
        """Initialize empty constraint set."""
        self.constraints: list[GeometricConstraint] = []

    def add(self, name: str, expression: Callable, description: str = ""):
        """Add a constraint.

        Args:
            name: Constraint name
            expression: Function returning violation amount
            description: Human-readable description
        """
        self.constraints.append(GeometricConstraint(name, expression, description))

    def add_min_value(self, param_name: str, min_value: float):
        """Add minimum value constraint for a parameter.

        Args:
            param_name: Parameter name
            min_value: Minimum allowed value
        """

        def check(params):
            return min_value - params.get(param_name, float("inf"))

        self.add(
            f"min_{param_name}",
            check,
            f"{param_name} must be >= {min_value}",
        )

    def add_max_value(self, param_name: str, max_value: float):
        """Add maximum value constraint for a parameter.

        Args:
            param_name: Parameter name
            max_value: Maximum allowed value
        """

        def check(params):
            return params.get(param_name, -float("inf")) - max_value

        self.add(
            f"max_{param_name}",
            check,
            f"{param_name} must be <= {max_value}",
        )

    def add_ratio_constraint(self, param_a: str, param_b: str, min_ratio: float):
        """Add constraint that param_b >= param_a * min_ratio.

        Args:
            param_a: First parameter name
            param_b: Second parameter name
            min_ratio: Minimum ratio b/a
        """

        def check(params):
            a = params.get(param_a, 0)
            b = params.get(param_b, 0)
            if a == 0:
                return 0  # Avoid division by zero
            return a * min_ratio - b

        self.add(
            f"{param_b}_over_{param_a}",
            check,
            f"{param_b}/{param_a} must be >= {min_ratio}",
        )

    def evaluate_all(self, params: dict) -> dict[str, float]:
        """Evaluate all constraints.

        Args:
            params: Dict of parameter name-value pairs

        Returns:
            Dict mapping constraint names to violation amounts
        """
        return {c.name: c.evaluate(params) for c in self.constraints}

    def is_feasible(self, params: dict) -> bool:
        """Check if all constraints are satisfied.

        Args:
            params: Dict of parameter name-value pairs

        Returns:
            True if all constraints satisfied (all violations == 0)
        """
        violations = self.evaluate_all(params)
        return all(v <= 1e-10 for v in violations.values())

    def total_violation(self, params: dict) -> float:
        """Compute total constraint violation.

        Args:
            params: Dict of parameter name-value pairs

        Returns:
            Sum of all violation amounts
        """
        return sum(self.evaluate_all(params).values())


class GeometryOptimizer:
    """Optimizer for parametric geometry.

    Wraps parametric geometry with objective functions and constraints
    for use with scipy.optimize or similar optimization algorithms.

    Args:
        geometry: Parametric primitive to optimize
        grid: Grid for voxelization
        constraints: Optional constraint set

    Example:
        >>> from strata_fdtd.parametric import ParametricHorn, GeometryOptimizer
        >>> from strata_fdtd import UniformGrid
        >>>
        >>> horn = ParametricHorn(throat_radius=0.02, mouth_radius=0.06, length=0.15)
        >>> grid = UniformGrid(shape=(100, 100, 150), resolution=2e-3)
        >>> optimizer = GeometryOptimizer(horn, grid)
        >>>
        >>> # Define objective (example: maximize volume)
        >>> def volume_objective(mask):
        ...     return -mask.sum()  # Negative because we minimize
        >>>
        >>> # Run optimization
        >>> result = optimizer.optimize(volume_objective)
        >>> print(f"Optimal: {result['parameters']}")
    """

    def __init__(
        self,
        geometry: ParametricPrimitive,
        grid: UniformGrid,
        constraints: ConstraintSet | None = None,
    ):
        """Initialize optimizer.

        Args:
            geometry: Parametric primitive to optimize
            grid: Grid for voxelization during evaluation
            constraints: Optional constraint set
        """
        self.geometry = geometry
        self.grid = grid
        self.constraints = constraints or ConstraintSet()
        self._param_names = [p.name for p in geometry.get_parameters()]
        self._param_bounds = [p.bounds for p in geometry.get_parameters()]

    def objective(
        self,
        params_vector: NDArray[np.floating],
        objective_fn: Callable[[NDArray[np.bool_]], float],
    ) -> float:
        """Evaluate objective function for parameter vector.

        Args:
            params_vector: 1D array of parameter values
            objective_fn: Function taking geometry mask, returning scalar

        Returns:
            Objective value (lower is better). Returns large penalty if infeasible.
        """
        # Convert vector to dict
        params = dict(zip(self._param_names, params_vector, strict=True))

        # Check constraints
        total_violation = self.constraints.total_violation(params)
        if total_violation > 1e-10:
            # Infeasible: return large penalty proportional to violation
            return 1e6 * (1 + total_violation)

        # Build geometry and voxelize
        try:
            geo = self.geometry.with_parameters(**params)
            mask = geo.voxelize(self.grid)
        except Exception:
            # Geometry construction failed: return large penalty
            return 1e9

        # Evaluate objective
        return objective_fn(mask)

    def optimize(
        self,
        objective_fn: Callable[[NDArray[np.bool_]], float],
        method: str = "Nelder-Mead",
        **scipy_options,
    ) -> dict:
        """Run optimization.

        Args:
            objective_fn: Objective function (mask -> scalar)
            method: scipy.optimize.minimize method
            **scipy_options: Additional options passed to scipy.optimize.minimize

        Returns:
            Dict with keys:
                - parameters: Optimal parameter values (dict)
                - objective: Optimal objective value
                - success: Whether optimization succeeded
                - geometry: Optimal geometry instance
                - message: Optimizer message
                - nfev: Number of function evaluations

        Raises:
            ImportError: If scipy is not installed
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError(
                "scipy is required for optimization. Install with: pip install scipy"
            ) from None

        # Initial values
        x0 = np.array([p.default for p in self.geometry.get_parameters()])

        # Run optimization
        result = minimize(
            lambda x: self.objective(x, objective_fn),
            x0,
            method=method,
            bounds=self._param_bounds,
            **scipy_options,
        )

        # Extract optimal parameters
        optimal_params = dict(zip(self._param_names, result.x, strict=True))

        return {
            "parameters": optimal_params,
            "objective": result.fun,
            "success": result.success,
            "geometry": self.geometry.with_parameters(**optimal_params),
            "message": result.message,
            "nfev": result.nfev,
        }
