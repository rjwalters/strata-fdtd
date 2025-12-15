# Builder Mode Guide

Builder Mode is a web-based visual editor for creating FDTD simulation scripts. It provides a Monaco code editor with Python syntax highlighting and a live 3D preview of your simulation geometry.

## Overview

Builder Mode helps you:
- **Write simulation scripts** with syntax highlighting and code completion
- **See your geometry** in real-time 3D preview as you type
- **Insert templates** for common patterns (sources, probes, materials)
- **Estimate resources** (memory, runtime) before downloading
- **Generate reproducible scripts** with content-based hashing

## Getting Started

### Launching Builder Mode

```bash
# Navigate to the viz directory
cd viz

# Install dependencies
pnpm install

# Start development server
pnpm dev

# Open http://localhost:5173 in your browser
```

**Note**: Builder Mode may also be available as a hosted web application at `https://fdtd-builder.example.com/` (check project documentation for current URL).

## Interface Overview

The Builder Mode interface consists of three main areas:

```
┌─────────────────────────────────────────────────────────┐
│  Simulation Builder              [Examples ▼] [Help]   │
├─────────────────────────┬───────────────────────────────┤
│ Script Editor           │ Live Preview                  │
│ ┌─────────────────────┐ │ ┌───────────────────────────┐ │
│ │ from strata_fdtd   │ │ │                           │ │
│ │ import FDTDSolver   │ │ │   [3D Scene]              │ │
│ │                     │ │ │   • Grid wireframe        │ │
│ │ solver=FDTDSolver(  │ │ │   • Material volumes      │ │
│ │   shape=(100,...),  │ │ │   • Source markers        │ │
│ │   resolution=1e-3   │ │ │   • Probe markers         │ │
│ │ )                   │ │ │                           │ │
│ │                     │ │ │   [Orbit Controls]        │ │
│ │ solver.add_source(  │ │ │   Rotate/Pan/Zoom         │ │
│ │   ...               │ │ │                           │ │
│ │ )                   │ │ │   [View Options]          │ │
│ │                     │ │ │   ☑ Grid  ☑ Materials    │ │
│ │                     │ │ │   ☑ Sources  ☑ Probes    │ │
│ │                     │ │ │   Slice: XY @ z=50 ▼      │ │
│ └─────────────────────┘ │ └───────────────────────────┘ │
│                         │                               │
│ [+ Source] [+ Probe]                                    │
│                         │ Estimated:                    │
│                         │ • Memory: 240 MB              │
│                         │ • Duration: 1.0 ms            │
│                         │ • Runtime: ~3 min @8 threads  │
├─────────────────────────┴───────────────────────────────┤
│ Script hash: abc123def456...                            │
│ [💾 Download Script] [📋 Copy Command]                  │
└─────────────────────────────────────────────────────────┘
```

### Script Editor (Left Panel)

**Features:**
- **Monaco Editor**: Full-featured code editor (same as VS Code)
- **Python syntax highlighting**: Color-coded keywords, strings, comments
- **Line numbers**: Easy reference and navigation
- **Minimap**: Overview of entire script
- **Undo/Redo**: Full edit history (Ctrl+Z, Ctrl+Y)
- **Auto-indentation**: Python-aware formatting

**Keyboard Shortcuts:**
- **Ctrl+S** (Cmd+S on Mac): Download script
- **Ctrl+/** (Cmd+/): Toggle line comment
- **Ctrl+F** (Cmd+F): Find
- **Ctrl+H** (Cmd+H): Find and replace
- **Alt+Shift+F**: Format code

### Live Preview (Right Panel)

**Features:**
- **3D rendering**: Real-time view of your simulation geometry
- **Interactive camera**: Rotate (drag), pan (right-drag), zoom (scroll)
- **Grid wireframe**: Shows simulation domain boundaries
- **Material volumes**: Semi-transparent colored regions
- **Source markers**: Sphere markers (color-coded by type)
- **Probe markers**: Small cube markers

**View Options:**
- **Toggle layers**: Show/hide grid, materials, sources, probes
- **Slice planes**: View XY, XZ, or YZ cross-sections at any position
- **Camera reset**: Return to default view angle

**Visual Encoding:**
- **Grid**: White wireframe box
- **Air**: Transparent (default material)
- **Solid materials**: Semi-transparent colored volumes
  - PZT (piezoelectric): Orange
  - Water: Blue
  - Aluminum: Gray
- **Sources**: Colored spheres
  - Gaussian pulse: Green
  - Sinusoidal: Yellow
- **Probes**: Red cubes

## Creating Your First Simulation

### Step 1: Start with a Template

Click the **Examples** dropdown and select **"Basic Pulse"**:

```python
from strata_fdtd import FDTDSolver, GaussianPulse

# Create solver with 100x100x100 grid (1mm resolution)
solver = FDTDSolver(
    shape=(100, 100, 100),
    resolution=1e-3
)

# Add Gaussian pulse source
solver.add_source(
    GaussianPulse(
        position=(0.025, 0.05, 0.05),
        frequency=40e3,
    )
)

# Add probe
solver.add_probe(
    "downstream",
    position=(0.075, 0.05, 0.05),
)
```

The 3D preview immediately shows:
- Grid bounding box (100mm cube)
- Green sphere at x=25mm (source)
- Red cube at x=75mm (probe)

### Step 2: Modify the Script

Try changing the source position. As you type, the preview updates (after 500ms delay):

```python
solver.add_source(
    GaussianPulse(
        position=(0.035, 0.05, 0.05),  # Changed from 0.025 to 0.035
        frequency=40e3,
    )
)
```

The green sphere marker moves to the new position instantly.

### Step 3: Review Estimates

The bottom panel shows:

- **Memory**: Estimated RAM usage (e.g., "240 MB")
- **Steps**: Number of timesteps based on CFL condition
- **Runtime**: Approximate execution time on your hardware

**Warning indicators:**
- ⚠ Yellow: Memory >4GB or runtime >5min
- 🔴 Red: Memory >8GB or runtime >10min
- Consider reducing grid size or using nonuniform grids

### Step 4: Download Script

Click **💾 Download Script**. The file is saved as:

```
simulation_abc123de.py
```

The filename includes a truncated SHA256 hash of the script content, ensuring:
- **Reproducibility**: Same script → same filename
- **Version tracking**: Different scripts → different filenames
- **Output matching**: Results are saved as `results_abc123de.h5`

### Step 5: Copy CLI Command

Click **📋 Copy Command** to copy:

```bash
fdtd-compute simulation_abc123de.py
```

Paste into your terminal to run the simulation.

## Using Template Buttons

Template buttons insert commonly-used code snippets. The cursor is positioned at the first parameter for easy editing.

### + Source

Inserts a source:

```python
solver.add_source(
    GaussianPulse(
        position=(|, , ),  # Cursor here
        frequency=
    )
)
```

**Source types:**
- `GaussianPulse`: Broadband pulse (good for impulse response)
- `Sinusoidal`: Single frequency tone (good for harmonic analysis)
- `Chirp`: Frequency sweep (good for bandwidth measurement)

### + Probe

Inserts a measurement probe:

```python
solver.add_probe(
    "|",  # Cursor here (probe name)
    position=(, , )
)
```

**Tips:**
- Use descriptive names: `"upstream"`, `"cavity"`, `"far_field"`
- Add multiple probes to compare different locations
- Probes add minimal computational cost

## Advanced Techniques

### Nonuniform Grids

Save memory while maintaining resolution where needed:

```python
from strata_fdtd import NonuniformGrid

# Fine resolution at center, coarse toward edges
grid = NonuniformGrid.from_stretch(
    shape=(100, 100, 100),
    base_resolution=1e-3,
    stretch_z=1.05,  # 5% growth per cell in z
    center_fine=True
)
```

The preview shows variable cell sizes (larger cells toward edges).

### Multiple Sources

Simulate phased arrays or interference:

```python
import numpy as np

# Two sources with 180° phase difference
solver.add_source(
    Sinusoidal(
        position=(0.03, 0.05, 0.05),
        frequency=40e3,
        phase=0
    )
)

solver.add_source(
    Sinusoidal(
        position=(0.07, 0.05, 0.05),
        frequency=40e3,
        phase=np.pi  # 180° out of phase
    )
)
```

The preview shows both source markers (both yellow for sinusoidal).

### Slice Planes

Useful for visualizing cross-sections of complex geometry:

In the preview panel:
1. Select slice plane: **XY**, **XZ**, or **YZ**
2. Adjust position slider to move through the volume
3. View 2D cross-section of materials and sources

Example use cases:
- Check if a sphere is properly centered
- Verify layer thicknesses in a multi-layer structure
- Debug unexpected geometry

## Best Practices

### Grid Sizing

**Rule of thumb**: Wavelength should span at least 10 cells.

```python
# For 40 kHz in air (λ ≈ 8.5mm):
# resolution < λ / 10 = 8.5mm / 10 = 0.85mm
# ✓ Good: resolution = 0.5mm (17 cells per wavelength)
# ✓ OK: resolution = 0.8mm (10.6 cells per wavelength)
# ✗ Bad: resolution = 1.0mm (8.5 cells per wavelength - marginal)
```

**Formula**:
```
resolution < c / (10 * f)

where:
  c = sound speed (343 m/s in air)
  f = highest frequency of interest
```

### Memory Estimation

**Formula**:
```
Memory (MB) ≈ N_cells * 8 bytes per cell / 1e6

Example:
  100³ grid = 1,000,000 cells = 8 MB
  200³ grid = 8,000,000 cells = 64 MB
```

**Tips**:
- Start small (100³) and scale up
- Use nonuniform grids for large domains
- Monitor estimates in the bottom panel

### Duration Selection

**How many timesteps do I need?**

```python
# Time for wave to cross the domain:
travel_time = domain_length / sound_speed

# Number of steps:
num_steps = travel_time / dt

# Example: 100mm domain, 343 m/s, dt ≈ 2ns
# travel_time = 0.1m / 343m/s ≈ 290µs
# num_steps = 290µs / 2ns ≈ 145,000 steps
```

**Rules of thumb**:
- **Pulse propagation**: 2-3× travel time
- **Resonance**: 10-20 periods of oscillation
- **Long-term dynamics**: Experiment and increase as needed

### Error Handling

Builder Mode shows errors in the script editor:

**Syntax errors**: Red underline with tooltip
```python
# Missing closing parenthesis
solver.add_source(
    GaussianPulse(position=(0.025, 0.05, 0.05), frequency=40e3
# ^ Red underline: SyntaxError
```

**Parse errors**: Preview shows "Waiting for valid script..."
- Fix syntax errors first
- Ensure required objects are defined (`solver`)

## Examples Gallery

Builder Mode includes pre-made examples accessible via the **Examples** dropdown:

| Example | Description | Grid Size |
|---------|-------------|-----------|
| **Basic Pulse** | Gaussian pulse in free field | 100³ |
| **PZT Transducer** | Piezoelectric array | 150³ |
| **Organ Pipes** | Helmholtz resonator array | 200³ |
| **Scattering** | Sphere in acoustic field | 128³ |
| **Waveguide** | Rectangular acoustic waveguide | 64×64×256 |
| **Nonuniform Grid** | Stretched grid demonstration | 100³ |

Click any example to load it into the editor.

## Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| **Ctrl+S** (Cmd+S) | Download script |
| **Ctrl+/** (Cmd+/) | Toggle line comment |
| **Ctrl+F** (Cmd+F) | Find |
| **Ctrl+H** (Cmd+H) | Find and replace |
| **Ctrl+Z** (Cmd+Z) | Undo |
| **Ctrl+Y** (Cmd+Y) | Redo |
| **Alt+Shift+F** | Format code |
| **Ctrl+Space** | Trigger autocomplete |

## Troubleshooting

### Preview not updating

**Problem**: Changes to script don't update the preview.

**Solutions:**
- Wait 500ms after typing (debounced update)
- Check for syntax errors (red underlines)
- Ensure `grid`, `scene`, and `solver` are defined
- Refresh page if preview is frozen

### Performance issues

**Problem**: Editor or preview is slow/laggy.

**Solutions:**
- Close other browser tabs
- Reduce grid size in preview (doesn't affect downloaded script)
- Disable minimap in editor settings
- Use a more powerful computer or reduce complexity

### Download not working

**Problem**: Download button doesn't save file.

**Solutions:**
- Check browser's download settings
- Ensure pop-ups are not blocked
- Try **Copy Command** instead and create file manually
- Check browser console for errors (F12 → Console tab)

## Further Reading

- **[Getting Started Guide](getting-started.md)** - Your first simulation
- **[CLI Reference](cli-reference.md)** - Running simulations from command line
- **[Viewer Mode Guide](viewer-guide.md)** - Visualizing results
- **[API Reference](api-reference.md)** - Complete Python API
- **[Troubleshooting](troubleshooting.md)** - Common problems and solutions

---

**Builder Mode Status**: This documentation describes the intended functionality of Builder Mode as specified in the project requirements. Some features may still be under development. Check the project repository for current implementation status.
