# App Icons

This directory should contain app icons for the FDTD Simulator desktop application.

## Generating Icons

To generate all required icon formats from a source PNG image:

```bash
npx @tauri-apps/cli icon path/to/icon-source.png
```

This will generate:
- `32x32.png` - Small icon
- `128x128.png` - Medium icon
- `128x128@2x.png` - Retina medium icon
- `icon.icns` - macOS app icon
- `icon.ico` - Windows app icon
- `icon.png` - Linux app icon

## Required Source Image

The source image should be:
- Square (1:1 aspect ratio)
- At least 512x512 pixels
- PNG format with transparency
- Simple, recognizable design that works at small sizes

## TODO

Create a proper app icon with a waveform or acoustic-related design that represents FDTD simulation.
