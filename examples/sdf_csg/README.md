# CSG Operations for Loudspeaker Design - Examples

This directory contains practical examples demonstrating how to use Constructive Solid Geometry (CSG) operations to model acoustic geometries for loudspeaker design and strata_fdtd structures.

## Overview

CSG (Constructive Solid Geometry) allows you to build complex 3D shapes by combining simple primitives using boolean operations:

- **Union**: Combine shapes (OR operation)
- **Intersection**: Overlap only (AND operation)
- **Difference**: Subtract shapes (NOT operation)
- **Smooth variants**: Blend shapes with smooth transitions

These examples show how to apply CSG to real acoustic design problems, from loudspeaker enclosures to metamaterials.

## Examples

### 1. `ported_enclosure.py` - Loudspeaker Enclosure with Bass Reflex Port

**What it demonstrates:**
- Hollow box construction (difference of two boxes)
- Driver cutout using cylinder subtraction
- Bass reflex port with flared mouth (union of cylinder + cone)
- Voxelization for FDTD acoustic simulation
- Helmholtz resonance frequency calculation

**Design:**
- 200mm × 200mm × 300mm enclosure
- 18mm wall thickness (typical MDF)
- 130mm driver diameter (5" woofer)
- 50mm port tuned to ~50 Hz

**Run it:**
```bash
python3 ported_enclosure.py
```

**Key concepts:**
- Creating hollow structures with `Difference`
- Combining multiple CSG operations in a tree
- Calculating port tuning frequency
- Voxelization for simulation

---

### 2. `helmholtz_resonator.py` - Helmholtz Resonator with Smooth Blending

**What it demonstrates:**
- Hollow spherical or cubic cavity
- Cylindrical neck opening
- Smooth transitions using `SmoothUnion`
- Comparison of smooth vs sharp CSG operations
- Theoretical resonance frequency analysis

**Physics:**
- Helmholtz resonators act as acoustic mass-spring systems
- Resonance frequency: f₀ = (c/2π) × sqrt(S/(V×L))
- Used in bass traps, absorbers, and metamaterials

**Run it:**
```bash
python3 helmholtz_resonator.py
```

**Key concepts:**
- `SmoothUnion` for realistic blended geometry
- End corrections for accurate frequency prediction
- Cavity volume and neck dimensions control tuning
- Comparison of smooth vs sharp operations

---

### 3. `metamaterial_unit_cell.py` - Acoustic Metamaterial Unit Cell

**What it demonstrates:**
- Periodic unit cell with central cavity
- Six connecting channels (±x, ±y, ±z directions)
- Array creation by translation
- Sub-wavelength structure design
- Metamaterial property analysis

**Applications:**
- Acoustic absorbers and barriers
- Phononic crystals
- Negative effective material properties
- Frequency-selective waveguides

**Run it:**
```bash
python3 metamaterial_unit_cell.py
```

**Key concepts:**
- Creating arrays of unit cells with `translate()`
- Sub-wavelength criterion (cell < λ/4)
- Periodic boundary conditions for simulation
- Coupled resonator systems

---

## Running the Examples

### Prerequisites

Ensure you have the strata_fdtd package installed:
```bash
pip install -e .
```

Optional dependencies for visualization:
```bash
pip install matplotlib
```

### Running Examples

Each example is a standalone script that can be run directly:

```bash
# Ported loudspeaker enclosure
python3 examples/sdf_csg/ported_enclosure.py

# Helmholtz resonator
python3 examples/sdf_csg/helmholtz_resonator.py

# Metamaterial unit cell
python3 examples/sdf_csg/metamaterial_unit_cell.py
```

### What to Expect

Each example will:
1. Print design parameters and theoretical predictions
2. Create the CSG geometry
3. Voxelize for FDTD simulation
4. Display cross-section visualization (if matplotlib available)
5. Print analysis and next steps

## Key CSG Operations

### Basic Operations

```python
from strata_fdtd.sdf import Box, Sphere, Cylinder, Cone
from strata_fdtd.sdf import Union, Intersection, Difference

# Create primitives
box = Box(center=(0, 0, 0), size=(0.1, 0.1, 0.1))
sphere = Sphere(center=(0, 0, 0), radius=0.05)

# Combine with CSG operations
combined = Union(box, sphere)  # OR: both shapes
overlap = Intersection(box, sphere)  # AND: only overlapping region
cut = Difference(box, sphere)  # NOT: box with sphere removed
```

### Smooth Operations

```python
from strata_fdtd.sdf import SmoothUnion, SmoothIntersection, SmoothDifference

# Blend shapes smoothly
blended = SmoothUnion(box, sphere, radius=0.01)  # 10mm blend radius
```

### Transformations

```python
# Translate, rotate, scale
translated = box.translate((0.1, 0, 0))
rotated = box.rotate_z(np.pi/4)  # 45° around z-axis
scaled = box.scale(2.0)  # 2x larger
```

### Voxelization

```python
from strata_fdtd.grid import UniformGrid

# Create voxelization grid
grid = UniformGrid(shape=(100, 100, 100), resolution=0.001)  # 1mm cells

# Convert geometry to boolean mask
voxel_mask = geometry.voxelize(grid)  # True = air, False = solid
```

## Design Workflow

### 1. Design Phase
- Sketch geometry with CSG primitives
- Calculate theoretical parameters (frequencies, volumes, etc.)
- Choose appropriate resolution for voxelization

### 2. Implementation
- Create primitives (Box, Sphere, Cylinder, Cone, Horn)
- Combine with CSG operations (Union, Intersection, Difference)
- Add smooth transitions where needed
- Translate/rotate/scale as required

### 3. Voxelization
- Create UniformGrid with appropriate resolution
- Voxelize geometry to boolean mask
- Verify air/solid cell counts

### 4. Simulation
- Use voxel mask with FDTDSolver
- Add sources and probes
- Run acoustic simulation
- Analyze results

## Tips and Best Practices

### Resolution Selection
- **Rule of thumb**: 10-20 cells per wavelength at highest frequency
- **Small features**: Ensure at least 3-5 cells across smallest dimension
- **Trade-off**: Higher resolution = better accuracy but more computation

### CSG Efficiency
- **Union scales well**: Can combine many primitives efficiently
- **Bounding boxes**: Used for optimization during voxelization
- **Lazy evaluation**: SDF only computed when needed

### Smooth vs Sharp Operations
- **Sharp operations**: Exact boolean logic, crisp edges
- **Smooth operations**: Realistic blending, no sharp corners
- **Blend radius**: Typically 5-10% of smallest feature size
- **Physics impact**: Smoothing slightly changes internal volumes

### Common Patterns

**Hollow structures:**
```python
outer = Box(center=(0, 0, 0), size=(0.1, 0.1, 0.1))
inner = Box(center=(0, 0, 0), size=(0.08, 0.08, 0.08))
shell = Difference(outer, inner)
```

**Arrays:**
```python
cells = [primitive.translate((i*spacing, 0, 0)) for i in range(n)]
array = Union(*cells)
```

**Tubes with flares:**
```python
tube = Cylinder(p1=(0, 0, 0), p2=(0, 0, 0.1), radius=0.01)
flare = Cone(p1=(0, 0, 0.1), p2=(0, 0, 0.15), r1=0.01, r2=0.02)
port = Union(tube, flare)
```

## Further Exploration

### Variations to Try

1. **Ported Enclosure:**
   - Different port lengths for tuning
   - Multiple ports for increased airflow
   - Exponential or hyperbolic horn flares
   - Offset driver positions

2. **Helmholtz Resonator:**
   - Cubic vs spherical cavities
   - Multiple necks for broadband absorption
   - Arrays of resonators at different frequencies
   - Tapered necks

3. **Metamaterial:**
   - Different cavity shapes (cubes, cylinders)
   - Varying channel dimensions
   - 2D vs 3D arrays
   - Gradient metamaterials (varying properties across array)

### Advanced Topics

- **Optimization**: Use CSG with parameter sweeps to optimize designs
- **Inverse design**: Start from desired frequency response, optimize geometry
- **Coupled systems**: Multiple resonators with shared walls
- **Graded structures**: Smoothly varying properties across geometry

## References

### Theory
- L. L. Beranek, "Acoustics" (1954) - Loudspeaker enclosure design
- M. Yang et al., "Acoustic metamaterials and phononic crystals" (2015)
- Inigo Quilez, "SDF Functions" - https://iquilezles.org/articles/distfunctions/

### Related Code
- `src/strata_fdtd/sdf.py` - SDF primitives implementation
- `tests/strata_fdtd/test_sdf_csg.py` - CSG operation tests
- `docs/CLAUDE.md` - Full project documentation

## Contributing

Found a bug or have an idea for a new example? Please open an issue or submit a pull request!

Potential new examples:
- Horn-loaded compression driver
- Transmission line enclosure
- Acoustic labyrinth
- Bandpass enclosure
- Metamaterial acoustic lens
- Phononic crystal waveguide
