# FDTD Visualizer

A 3D visualization tool for FDTD (Finite-Difference Time-Domain) acoustic simulations.

## Features

- **3D Voxel Rendering**: Visualize pressure fields with diverging colormaps
- **Geometry Overlay**: Display material boundaries (air/solid)
- **Time Series Plots**: Interactive probe data visualization
- **Spectrum Analysis**: FFT magnitude and spectrogram views
- **Flow Particles**: Animate acoustic velocity fields
- **Export**: Screenshots, GIF animations, and data export

## Quick Start

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Run tests
pnpm test

# Build for production
pnpm build
```

## Loading Simulation Data

The visualizer loads simulation data from a directory containing:

- `manifest.json` - Simulation metadata and file references
- `metadata.json` - Grid dimensions, resolution, timestep info
- `probes.json` - Probe time series data
- `geometry.json` - Material boundaries (air/solid)
- `snapshots/` - Binary pressure field snapshots (`.bin`)

### Loading via URL

```
http://localhost:5173/?data=demos/organ-pipes
```

### URL Parameters

Viewer state is encoded in URL parameters for sharing:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `data` | Path to simulation data | `demos/organ-pipes` |
| `frame` | Current frame index | `42` |
| `speed` | Playback speed (0.25-4) | `2` |
| `cmap` | Colormap (`diverging`, `magnitude`, `viridis`) | `viridis` |
| `thresh` | Pressure threshold (0-1) | `0.1` |
| `geom` | Voxel geometry (`point`, `mesh`, `hidden`) | `mesh` |
| `probes` | Selected probe names | `upstream,cavity` |

### Export Capabilities

- **Screenshots**: PNG capture at customizable resolution
- **Animations**: GIF export of frame sequences
- **Data Export**: JSON (full state) or CSV (probe data)

## Performance Benchmarks

The visualizer includes automated performance benchmarks to measure rendering performance at various grid sizes.

### Running Benchmarks

```bash
# Run with default settings (headless)
pnpm benchmark

# Watch the benchmark run in browser
pnpm benchmark:visible

# Custom configuration
pnpm benchmark --grid-sizes 50,100,150 --frames 60

# Output to specific file
pnpm benchmark --output results.json
```

### Benchmark Options

| Option | Default | Description |
|--------|---------|-------------|
| `-o, --output <file>` | `benchmark-results.json` | Output JSON file |
| `-g, --grid-sizes <sizes>` | `50,100,150,200` | Comma-separated grid sizes |
| `-f, --frames <count>` | `120` | Frames to collect per test |
| `-w, --warmup <count>` | `30` | Warmup frames before collecting |
| `--no-downsample` | - | Disable downsampling tests |
| `--target-voxels <count>` | `262144` | Target voxels for downsampling |
| `--headless <bool>` | `true` | Run headless browser |
| `-p, --port <number>` | `5174` | Dev server port |

### Interpreting Results

The benchmark outputs a JSON file with the following structure:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "platform": {
    "userAgent": "...",
    "hardwareConcurrency": 8,
    "devicePixelRatio": 2,
    "renderer": "Apple M1 Pro"
  },
  "config": { ... },
  "results": [
    {
      "gridSize": 100,
      "voxelCount": 1000000,
      "downsampled": false,
      "renderedVoxels": 1000000,
      "fps": {
        "mean": 45.2,
        "min": 38.1,
        "max": 52.3,
        "p50": 45.0,
        "p95": 41.2,
        "p99": 39.5
      },
      "frameTime": {
        "mean": 22.1,
        "min": 19.1,
        "max": 26.2,
        "p50": 22.2,
        "p95": 24.3,
        "p99": 25.3
      },
      "memoryMB": 11.44,
      "timeToFirstFrame": 45.2
    }
  ]
}
```

### Performance Targets

The following performance targets are tracked:

| Grid Size | Target FPS | Notes |
|-----------|------------|-------|
| 100³ | 30+ FPS | Smooth real-time playback |
| 150³ | 15+ FPS | Acceptable playback |

Results are color-coded in the terminal output:
- **Green**: >= 30 FPS
- **Yellow**: 15-30 FPS
- **Red**: < 15 FPS

### Manual Benchmarking

You can also access the benchmark page directly in the browser:

```
http://localhost:5173/benchmark.html
```

URL parameters configure the benchmark:
- `gridSizes=50,100,150` - Grid sizes to test
- `frames=120` - Frames per test
- `warmup=30` - Warmup frames
- `downsample=true` - Test downsampling
- `targetVoxels=262144` - Target voxels for downsampling

## Deployment (Cloudflare Pages)

The visualizer is configured for deployment on Cloudflare Pages.

### Prerequisites

1. Create a Cloudflare account at https://dash.cloudflare.com
2. Authenticate wrangler: `pnpm exec wrangler login`

### Deploy

```bash
# Build and deploy
pnpm build
pnpm pages:deploy

# Or deploy with automatic project creation
pnpm pages:deploy
```

On first deploy, wrangler will create the `fdtd-viz` project in your Cloudflare account.

### Local Preview (with Cloudflare environment)

```bash
# Build first
pnpm build

# Preview with Cloudflare Pages environment (includes _headers support)
pnpm pages:preview
```

### CI/CD Setup

For GitHub Actions, add these secrets to your repository:
- `CLOUDFLARE_API_TOKEN` - API token with Pages permissions
- `CLOUDFLARE_ACCOUNT_ID` - Your Cloudflare account ID

Example workflow:
```yaml
- name: Deploy to Cloudflare Pages
  run: |
    cd viz
    pnpm install
    pnpm build
    pnpm exec wrangler pages deploy dist --project-name fdtd-viz
  env:
    CLOUDFLARE_API_TOKEN: ${{ secrets.CLOUDFLARE_API_TOKEN }}
    CLOUDFLARE_ACCOUNT_ID: ${{ secrets.CLOUDFLARE_ACCOUNT_ID }}
```

### Headers Configuration

The site includes special headers for FFmpeg.wasm SharedArrayBuffer support:
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

These are configured in `public/_headers` and automatically deployed.

## Development

### Project Structure

```
viz/
├── src/
│   ├── benchmark/          # Benchmark components
│   │   ├── BenchmarkRunner.tsx
│   │   └── main.tsx
│   ├── components/         # React components
│   ├── lib/               # Utilities
│   │   └── performance.ts # Performance tracking
│   ├── stores/            # Zustand stores
│   └── App.tsx
├── scripts/
│   └── benchmark.ts       # Benchmark CLI script
├── index.html             # Main app entry
├── benchmark.html         # Benchmark entry
└── vite.config.ts
```

### Performance Optimization

The visualizer uses several techniques for optimal rendering:

1. **Adaptive Downsampling**: Automatically reduces grid resolution for large datasets
2. **Instanced Rendering**: Uses Three.js InstancedMesh for efficient voxel rendering
3. **Threshold Filtering**: Only renders voxels above a configurable threshold
4. **Display Fill Control**: Randomly hides voxels to reduce visual density

See `src/lib/performance.ts` for implementation details.
