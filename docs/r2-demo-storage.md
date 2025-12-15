# R2 Demo Storage

This document describes how to set up and manage Cloudflare R2 storage for hosting demo simulation data for the Strata web viewer.

## Overview

Demo simulation data is stored in Cloudflare R2 object storage to:
- Avoid deploying large binary files with the web application
- Enable fast, global delivery via Cloudflare's edge network
- Reduce git repository size and clone times

## Architecture

```
strata-demos (R2 bucket)
└── demos/
    └── organ-pipes/
        ├── open-pipe/
        │   ├── open-pipe_manifest.json
        │   ├── open-pipe_metadata.json
        │   ├── open-pipe_probes.json
        │   ├── open-pipe_geometry.bin
        │   └── open-pipe_snapshot_*.bin
        ├── closed-pipe/
        │   └── ...
        └── half-open-pipe/
            └── ...
```

## Prerequisites

1. **Cloudflare Account** with R2 enabled
2. **Wrangler CLI** installed and authenticated:
   ```bash
   npm install -g wrangler
   wrangler login
   ```

## Initial Setup

### 1. Create the R2 Bucket

From the `strata-web` directory:

```bash
# Run the setup script
./scripts/setup-r2.sh
```

Or manually:

```bash
wrangler r2 bucket create strata-demos
```

### 2. Enable Public Access

1. Go to Cloudflare Dashboard > R2 > `strata-demos` > Settings
2. Under **Public access**, click **Allow Access**
3. Choose one of:
   - **R2.dev subdomain**: Get a URL like `https://pub-<id>.r2.dev`
   - **Custom domain**: Connect your own domain

4. Note the public URL for configuration

### 3. Configure CORS (if using custom domain)

For custom domains, you may need to configure CORS. The R2.dev subdomain includes permissive CORS headers by default.

## Uploading Demo Data

### Upload All Demos

```bash
cd strata-web
./scripts/upload-demos-to-r2.sh
```

### Upload a Specific Demo

```bash
./scripts/upload-demos-to-r2.sh closed-pipe
```

### Manual Upload

For individual files:

```bash
wrangler r2 object put strata-demos/demos/organ-pipes/closed-pipe/manifest.json \
  --file=../shared/demos/organ-pipes/closed-pipe/closed-pipe_manifest.json \
  --content-type=application/json
```

## Configuration

### Environment Variables

Create a `.env` file in `strata-web/`:

```bash
# Copy the example
cp .env.example .env

# Edit with your bucket URL
VITE_R2_BUCKET_URL=https://pub-xxxxx.r2.dev
```

### Wrangler Configuration

The `wrangler.toml` includes an R2 bucket binding:

```toml
[[r2_buckets]]
binding = "DEMOS_BUCKET"
bucket_name = "strata-demos"
```

This binding can be used by Cloudflare Pages Functions to access the bucket programmatically if needed.

## Demo Data Format

Each demo directory contains:

| File | Description |
|------|-------------|
| `*_manifest.json` | Index of all files and snapshot metadata |
| `*_metadata.json` | Simulation parameters and grid info |
| `*_probes.json` | Probe location and time series data |
| `*_geometry.bin` | Boundary geometry as binary data |
| `*_snapshot_*.bin` | Pressure field snapshots (float16) |

### Manifest Format

```json
{
  "metadata": "demo_metadata.json",
  "probes": "demo_probes.json",
  "geometry": "demo_geometry.bin",
  "snapshots": [
    {
      "shape": [130, 44, 44],
      "dtype": "float16",
      "format": "float16",
      "downsample": 1,
      "file": "demo_snapshot_0000.bin",
      "bytes": 503360,
      "time": 0.0
    }
  ]
}
```

## Adding New Demos

1. **Generate demo data** using the Strata CLI:
   ```bash
   strata run my-simulation.py --output ./output
   strata export ./output --format web --dest ../shared/demos/my-demo
   ```

2. **Upload to R2**:
   ```bash
   cd strata-web
   ./scripts/upload-demos-to-r2.sh my-demo
   ```

3. **Update the gallery** in `src/pages/ExamplesGallery.tsx`:
   ```typescript
   const DEMOS: Demo[] = [
     // ... existing demos
     {
       id: 'my-demo',
       title: 'My New Demo',
       description: 'Description of the simulation',
       category: 'Acoustics',
       path: '/demos/category/my-demo',
     },
   ]
   ```

## Troubleshooting

### Upload Fails

1. Check authentication: `wrangler whoami`
2. Verify bucket exists: `wrangler r2 bucket list`
3. Check file permissions and paths

### CORS Errors

If you see CORS errors in the browser console:

1. Ensure public access is enabled on the bucket
2. For custom domains, verify CORS settings in Cloudflare dashboard
3. Clear browser cache and retry

### Files Not Found (404)

1. Verify the file was uploaded: `wrangler r2 object list strata-demos --prefix demos/`
2. Check the URL path matches the upload path
3. Ensure public access is enabled

## Cost Considerations

R2 pricing (as of 2024):
- **Storage**: $0.015/GB/month
- **Class A operations** (writes): $4.50/million
- **Class B operations** (reads): $0.36/million
- **Egress**: Free (no bandwidth charges)

For ~5.5GB of demo data with moderate read traffic, expect costs under $1/month.

## Security Notes

- Demo data is public and read-only
- Do not store sensitive simulation data in this bucket
- The R2 bucket binding in wrangler.toml does not expose write access to the web application
