# Viewer Mode Guide

Viewer Mode is a web-based 3D visualization tool for exploring FDTD simulation results. It provides interactive playback, probe analysis, and export capabilities.

## Overview

Viewer Mode helps you:
- **Visualize pressure fields** in 3D with customizable colormaps
- **Animate wave propagation** with playback controls
- **Analyze probe data** with time series and FFT plots
- **Explore geometry** with material boundary overlays
- **Export results** as screenshots, GIF animations, or data files

## Getting Started

### Installation

```bash
# Navigate to the viz directory
cd viz

# Install dependencies (requires pnpm)
pnpm install

# Start development server
pnpm dev
```

The visualizer will be available at **http://localhost:5173**.

### Loading Simulation Results

#### Method 1: Upload HDF5 File

1. Run your simulation to generate `results.h5`
2. Open the Viewer in your browser
3. Click **"Upload Results"** button
4. Select your `results.h5` file
5. Wait for upload and parsing (progress shown in bottom-right)

#### Method 2: Load from URL (Local Development)

Place your results file in `viz/public/demos/` and access via URL parameter:

```
http://localhost:5173/?data=demos/my_results
```

The visualizer will load `viz/public/demos/my_results/` which should contain:
- `manifest.json`
- `metadata.json`
- `probes.json`
- `geometry.json` (optional)
- `snapshots/*.bin` (binary snapshot files)

**Example directory structure:**
```
viz/public/demos/organ-pipes/
├── manifest.json       # File references and metadata
├── metadata.json       # Grid dimensions, resolution, timestep
├── probes.json         # Probe time series data
├── geometry.json       # Material boundaries (optional)
└── snapshots/
    ├── 0.bin
    ├── 100.bin
    ├── 200.bin
    └── ...
```

## Interface Overview

```
┌─────────────────────────────────────────────────────────┐
│ FDTD Viewer                        [Upload] [Export ▼] │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                   3D Visualization                      │
│                                                         │
│          [Pressure field with colormap]                 │
│                                                         │
│          Rotate: Left-drag                              │
│          Pan: Right-drag                                │
│          Zoom: Scroll                                   │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ ◄◄  ◄  ▶  ▶▶  [=========|======]  Frame: 42/100       │
│ Loop: ☑  Speed: [====|=====] 1.0x                      │
├─────────────────────────────────────────────────────────┤
│ View Options                                            │
│ Colormap: [Diverging ▼]  Threshold: [===|====] 0.1    │
│ Geometry: [Point ▼]      Display Fill: [==|=====] 0.3  │
│ Slice: [XY ▼] @ [====|====] 50                        │
├─────────────────────────────────────────────────────────┤
│ Probe Analysis                                          │
│ Selected: [downstream ▼]                                │
│ ┌─────────────────────────┬─────────────────────────┐  │
│ │ Time Series             │ FFT Spectrum            │  │
│ │ [Line plot]             │ [Frequency plot]        │  │
│ └─────────────────────────┴─────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Playback Controls

### Play/Pause

- **▶ Play**: Start animation (advances one frame per render)
- **❚❚ Pause**: Stop animation
- **Spacebar**: Toggle play/pause (keyboard shortcut)

### Frame Navigation

- **◄◄ First**: Jump to frame 0
- **◄ Previous**: Go back one frame (or **Left Arrow**)
- **▶ Next**: Advance one frame (or **Right Arrow**)
- **▶▶ Last**: Jump to final frame

### Scrubber

Drag the slider to jump to any frame:
- Click to jump
- Drag for continuous scrubbing
- Displays current frame number and total

### Loop Mode

- **☑ Loop**: Auto-restart when reaching the end
- **☐ No Loop**: Stop at final frame

### Playback Speed

Adjust animation speed from 0.25x to 4x:
- **0.25x**: Slow motion (4× slower than real-time)
- **0.5x**: Half speed
- **1.0x**: Real-time (default)
- **2.0x**: Double speed
- **4.0x**: Fast forward

**Note**: "Real-time" means one frame per render cycle (~60 FPS if simulation timesteps allow).

## 3D Visualization

### Camera Controls

**Mouse:**
- **Left-drag**: Rotate camera around scene
- **Right-drag** (or **Ctrl+Left-drag**): Pan camera
- **Scroll**: Zoom in/out
- **Double-click**: Reset camera to default view

**Keyboard:**
- **R**: Reset camera
- **F**: Frame all (zoom to fit)

### Colormaps

Three colormap options for pressure field visualization:

#### 1. Diverging (Red-Blue)
**Best for**: Bipolar pressure fields (positive and negative pressure)

- **Blue**: Negative pressure (rarefaction)
- **White**: Zero pressure
- **Red**: Positive pressure (compression)

**Example**: Visualizing sinusoidal waves, reflections, interference

#### 2. Magnitude
**Best for**: Absolute pressure magnitude (ignores sign)

- **Dark**: Low magnitude
- **Bright**: High magnitude

**Example**: Visualizing wave amplitude, energy distribution

#### 3. Viridis (Perceptually Uniform)
**Best for**: Grayscale-safe visualization, accessibility

- **Purple**: Low values
- **Green**: Medium values
- **Yellow**: High values

**Example**: Publications, colorblind-friendly displays

### Threshold Control

Filter out low-amplitude voxels to reduce visual clutter:

- **0.0**: Show all voxels (default)
- **0.1**: Hide voxels with pressure < 10% of max
- **0.5**: Only show voxels with pressure > 50% of max
- **1.0**: Only show maximum pressure voxels

**Use case**: Highlighting wave fronts, reducing noise

### Display Fill Control

Randomly hide a fraction of voxels to see interior structure:

- **0.0**: Show every voxel (dense, opaque)
- **0.3**: Show 30% of voxels (default, balanced)
- **0.5**: Show 50% of voxels
- **1.0**: Show all voxels (same as 0.0)

**Use case**: Seeing through pressure field to geometry inside

## Geometry Display

Visualize material boundaries (if available in results):

### Voxel Geometry Modes

#### 1. Point
Show geometry as individual voxels (fastest):
- **Pro**: Fast rendering, low memory
- **Con**: Sparse appearance

#### 2. Mesh (Recommended)
Show geometry as connected mesh:
- **Pro**: Solid appearance, clear boundaries
- **Con**: Slightly slower than points

#### 3. Hidden
Don't show geometry (only pressure field):
- **Pro**: Fastest, focus on pressure
- **Con**: No context for material locations

### Material Colors

- **Air**: Transparent (not shown)
- **Solid materials**: Semi-transparent colored volumes
  - PZT: Orange
  - Water: Blue
  - Aluminum: Gray
  - Custom: Auto-assigned colors

## Slice Planes

View 2D cross-sections of the 3D pressure field:

### Slice Orientations

- **XY**: Horizontal slice (top-down view)
- **XZ**: Vertical slice (side view, Y-axis)
- **YZ**: Vertical slice (side view, X-axis)
- **None**: Show full 3D volume (default)

### Slice Position

Drag slider to move slice plane through the volume:
- **Min**: One edge of domain
- **Max**: Opposite edge of domain

**Use case**: Examining cross-sections, finding symmetry planes, debugging geometry

## Probe Analysis

Analyze pressure measurements at probe locations.

### Selecting a Probe

Use the **Selected** dropdown to choose a probe by name:
- Probes are named in the simulation script
- Example names: `"upstream"`, `"downstream"`, `"cavity"`

### Time Series Plot

**Left panel**: Pressure vs. time

**Features:**
- **Vertical line**: Current playback time
- **Zoom**: Click and drag to zoom into region
- **Reset zoom**: Double-click plot
- **Pan**: Shift+drag

**Interpretation:**
- **Arrival time**: When does the pulse reach the probe?
- **Amplitude**: Peak pressure value
- **Decay**: How does pressure decrease over time?
- **Reflections**: Secondary peaks indicate reflections

**Example observations:**
- Pulse arrives at ~600 steps (~1.2 µs for 50mm distance)
- Peak amplitude ~1.0 Pa
- Ringing indicates reflections from boundaries

### FFT Spectrum Plot

**Right panel**: Frequency content

**Features:**
- **X-axis**: Frequency (Hz or kHz)
- **Y-axis**: Magnitude (linear or log scale)
- **Peak markers**: Identify dominant frequencies

**Interpretation:**
- **Peak frequency**: Should match source frequency
- **Harmonics**: Integer multiples of fundamental
- **Bandwidth**: Spread of energy across frequencies

**Example observations:**
- Peak at 40 kHz (matches source)
- Bandwidth ~10 kHz (Gaussian pulse is broadband)
- Higher harmonics indicate nonlinear effects or reflections

### Export Probe Data

Click **Export** → **Probe Data** to download:

**CSV format:**
```csv
time,pressure
0.0,0.0
1.94e-09,0.023
3.88e-09,0.15
...
```

**JSON format:**
```json
{
  "probe_name": "downstream",
  "position": [0.075, 0.05, 0.05],
  "time": [0.0, 1.94e-09, ...],
  "pressure": [0.0, 0.023, ...]
}
```

## Export Capabilities

### Screenshots

Capture current view as PNG image:

1. Navigate to desired frame and angle
2. Click **Export** → **Screenshot**
3. Choose resolution:
   - **1x**: Current browser resolution
   - **2x**: Double resolution (high quality)
   - **4x**: Quadruple resolution (publication quality)

**Output**: `fdtd_screenshot_frame_042.png`

### Animated GIF

Export frame sequence as animated GIF:

1. Set frame range (e.g., frames 0-100)
2. Click **Export** → **Animated GIF**
3. Configure:
   - **Frame rate**: 10 FPS (default), 30 FPS, 60 FPS
   - **Quality**: Low/Medium/High
   - **Loop**: ✓ Yes or ☐ No
4. Wait for encoding (uses FFmpeg.wasm in browser)

**Output**: `fdtd_animation.gif`

**Performance:**
- Small grids (<100³): ~10 seconds
- Large grids (>200³): May take minutes
- Reduce frame rate or quality for faster export

### Data Export

Export full state or probe data:

#### JSON (Full State)
Complete visualization state for reproducibility:
```json
{
  "frame": 42,
  "colormap": "diverging",
  "threshold": 0.1,
  "camera": { "position": [...], "target": [...] },
  "probes": { ... },
  "metadata": { ... }
}
```

#### CSV (Probe Data)
Probe time series for external analysis (Excel, Python, MATLAB).

## URL State Sharing

Viewer state is encoded in URL for easy sharing and bookmarking.

**Example URL:**
```
http://localhost:5173/?data=demos/organ-pipes&frame=42&speed=2&cmap=diverging&thresh=0.1&geom=mesh&probes=upstream,cavity
```

**Parameters:**
| Parameter | Description | Example |
|-----------|-------------|---------|
| `data` | Path to simulation data | `demos/organ-pipes` |
| `frame` | Current frame index | `42` |
| `speed` | Playback speed | `2` (2x) |
| `cmap` | Colormap | `diverging`, `magnitude`, `viridis` |
| `thresh` | Pressure threshold | `0.1` (10%) |
| `geom` | Geometry mode | `point`, `mesh`, `hidden` |
| `probes` | Selected probes | `upstream,cavity` |

**Use cases:**
- Share specific view with collaborators
- Bookmark interesting frames
- Embed in documentation or presentations

## Performance Optimization

The visualizer includes adaptive performance features:

### Automatic Downsampling

For large grids (>200³), the visualizer automatically reduces voxel count:

**Target**: ~262,000 voxels (configurable)

**Method**: Skip voxels uniformly to reduce density

**Example**:
- 300³ grid = 27M voxels → downsample to ~260k voxels
- Reduction: ~100x fewer voxels
- Visual impact: Slight reduction in detail, much faster rendering

**Override**: Use `--no-downsample` in benchmark mode to disable.

### Display Fill

Reduce visual density without changing data:
- Randomly hide voxels for "x-ray" view
- Adjustable from 0% to 100% fill
- No performance impact on rendering

### Reduce Frame Rate

If experiencing lag:
- Lower playback speed to 0.5x or 0.25x
- Skip frames manually using ▶ Next button
- Close other browser tabs to free resources

## Troubleshooting

### Slow Performance

**Symptoms**: Laggy rotation, low FPS, stuttering playback

**Solutions:**
1. **Reduce threshold**: Hide low-amplitude voxels (threshold 0.2+)
2. **Reduce display fill**: Show only 20-30% of voxels
3. **Use Point geometry**: Faster than Mesh mode
4. **Close other tabs**: Free up GPU/RAM
5. **Reduce grid size**: Re-run simulation with coarser resolution
6. **Use a more powerful GPU**: Integrated graphics may struggle with large grids

**Performance targets:**
| Grid Size | Target FPS | Notes |
|-----------|------------|-------|
| 100³      | 30+ FPS    | Smooth on most hardware |
| 150³      | 15+ FPS    | Acceptable playback |
| 200³      | 10 FPS     | Use downsampling |

### Upload Fails

**Symptoms**: "Failed to load results" error after upload

**Solutions:**
1. **Check file format**: Must be HDF5 (`.h5` or `.hdf5`)
2. **Check file size**: Browser upload limit is ~2GB
3. **Check file structure**: Must contain required datasets (see HDF5 format)
4. **Re-run simulation**: File may be corrupted
5. **Use local file server**: For very large files, use URL loading instead

### Probes Not Showing

**Symptoms**: Probe dropdown is empty or probe plots are blank

**Solutions:**
1. **Check script**: Ensure probes were added with `scene.add_probe(...)`
2. **Check names**: Probes must have names: `scene.add_probe(..., name="probe1")`
3. **Re-run simulation**: Old results may not include probe data
4. **Check HDF5 file**: Verify `/probes/` group exists with `h5dump results.h5`

### Geometry Not Visible

**Symptoms**: Material boundaries don't appear

**Solutions:**
1. **Check geometry mode**: Ensure not set to "Hidden"
2. **Check simulation**: Materials must be added in script
3. **Adjust camera**: May need to rotate to see geometry
4. **Check transparency**: Geometry is semi-transparent, may be hard to see

## Advanced Features

### Flow Particles (Future)

Animate acoustic velocity fields with particle traces:
- Particles follow velocity field
- Visualize acoustic streaming
- Adjustable particle count and lifetime

**Status**: Planned feature (not yet implemented)

### Spectrogram View (Future)

Time-frequency analysis of probe data:
- STFT (Short-Time Fourier Transform)
- Adjustable window size
- Colormap for magnitude

**Status**: Planned feature (not yet implemented)

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Spacebar** | Play/Pause |
| **Left Arrow** | Previous frame |
| **Right Arrow** | Next frame |
| **Home** | First frame |
| **End** | Last frame |
| **R** | Reset camera |
| **F** | Frame all (fit to view) |
| **1** | Diverging colormap |
| **2** | Magnitude colormap |
| **3** | Viridis colormap |

## Further Reading

- **[Getting Started Guide](getting-started.md)** - Your first simulation
- **[Builder Mode Guide](builder-guide.md)** - Visual script editor
- **[CLI Reference](cli-reference.md)** - Command-line simulation tool
- **[API Reference](api-reference.md)** - Python API
- **[Troubleshooting](troubleshooting.md)** - Common problems

## Developer Information

For technical details about the visualizer implementation:

- **Source**: `viz/src/` directory
- **README**: `viz/README.md` (technical documentation)
- **Performance Benchmarks**: `pnpm benchmark` (see viz/README.md)
- **Tests**: `pnpm test`

---

**Visualizer Documentation**: This guide describes the FDTD Viewer as it exists in the `viz/` directory. For the latest features and updates, see `viz/README.md`.
