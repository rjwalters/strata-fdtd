# FDTD Simulator Desktop App

This directory contains the Tauri-based desktop application wrapper for the FDTD Simulator web UI.

## Features

- **Native File Dialogs**: Open HDF5 simulation results using your system's native file picker
- **Drag and Drop**: Drop HDF5 files directly onto the app window to open them
- **Cross-Platform**: Supports macOS, Windows, and Linux
- **Small Bundle Size**: ~10-15 MB installer (vs 100+ MB with Electron)
- **System Integration**: System tray icon with quick access menu
- **Offline Usage**: Run the app without an internet connection

## Development

### Prerequisites

- Node.js 18+ and pnpm
- Rust 1.77.2+ (install from https://rustup.rs)
- Platform-specific dependencies:
  - **macOS**: Xcode Command Line Tools
  - **Windows**: Microsoft Visual C++ Build Tools
  - **Linux**: `webkit2gtk`, `libayatana-appindicator3`

### Running in Development Mode

```bash
cd viz
pnpm install
pnpm tauri:dev
```

This will:
1. Start the Vite dev server on http://localhost:5173
2. Compile the Rust backend
3. Launch the desktop app with hot-reload enabled

### Building for Production

```bash
# Build for current platform
pnpm tauri:build

# Build for specific platforms
pnpm tauri:build:macos    # Universal binary (Intel + Apple Silicon)
pnpm tauri:build:windows  # Windows .msi installer
pnpm tauri:build:linux    # Linux AppImage
```

Built packages will be in `src-tauri/target/release/bundle/`.

## Architecture

### Rust Backend (`src-tauri/`)

The Rust backend provides:
- Native file picker integration
- Event system for file opens and navigation
- System tray menu (future)
- Auto-update mechanism (future)

Key files:
- `src-tauri/src/lib.rs` - Main Tauri app logic and commands
- `src-tauri/src/main.rs` - Binary entry point
- `src-tauri/tauri.conf.json` - App configuration
- `src-tauri/Cargo.toml` - Rust dependencies

### Frontend Integration

The web UI detects the Tauri environment and adapts:
- `src/hooks/useTauri.ts` - React hooks for Tauri features
- `isTauri` flag - Conditional rendering for desktop-only features
- Event listeners - Handle file opens from system tray, etc.

Usage example:

```typescript
import { useTauri, useTauriFileOpen } from './hooks/useTauri';

function App() {
  const { isTauri, openFileDialog } = useTauri();
  
  // Listen for files opened via system tray
  useTauriFileOpen((path) => {
    loadSimulation(path);
  });
  
  const handleOpen = async () => {
    if (isTauri) {
      const path = await openFileDialog();
      if (path) loadSimulation(path);
    } else {
      // Fall back to web file input
      openWebFilePicker();
    }
  };
  
  return (
    <button onClick={handleOpen}>
      Open Simulation
    </button>
  );
}
```

## Tauri Commands

The Rust backend exposes these commands to the frontend:

### `open_file_dialog()`

Opens a native file picker dialog filtered to `.h5` and `.hdf5` files.

```typescript
const path = await invoke<string | null>('open_file_dialog');
```

### `open_recent_file(path: string)`

Navigates to the viewer and loads the specified file.

```typescript
await invoke('open_recent_file', { path: '/path/to/simulation.h5' });
```

### `navigate(route: string)`

Navigates to a specific route in the app.

```typescript
await invoke('navigate', { route: '/builder' });
```

## Events

The app can emit and listen to these events:

### `open-file`

Emitted when a file should be opened (e.g., from system tray).

```typescript
useTauriFileOpen((path: string) => {
  console.log('Open file:', path);
});
```

### `navigate`

Emitted when the app should navigate to a route.

```typescript
useTauriNavigate((route: string) => {
  navigate(route);
});
```

## Configuration

### App Metadata

Edit `src-tauri/tauri.conf.json`:
- `productName` - Display name of the app
- `version` - App version (should match `package.json`)
- `identifier` - Unique bundle identifier (e.g., `com.fdtd.simulator`)

### Window Settings

```json
{
  "app": {
    "windows": [{
      "title": "FDTD Simulator",
      "width": 1400,
      "height": 900,
      "resizable": true
    }]
  }
}
```

### File System Access

Configure allowed directories:

```json
{
  "plugins": {
    "fs": {
      "scope": [
        { "path": "$HOME/**/*.h5" },
        { "path": "$DOCUMENT/**/*.h5" }
      ]
    }
  }
}
```

## Distribution

### Code Signing

For macOS and Windows distribution, you'll need to sign the app:

**macOS**:
1. Get an Apple Developer account
2. Create a Developer ID Application certificate
3. Set environment variables:
   ```bash
   export APPLE_CERTIFICATE="Developer ID Application: Your Name"
   export APPLE_ID="your@email.com"
   export APPLE_PASSWORD="app-specific-password"
   ```

**Windows**:
1. Get a code signing certificate
2. Configure in `tauri.conf.json`:
   ```json
   {
     "bundle": {
       "windows": {
         "certificateThumbprint": "YOUR_THUMBPRINT",
         "digestAlgorithm": "sha256"
       }
     }
   }
   ```

### GitHub Releases

Automated builds can be set up using GitHub Actions. See `.github/workflows/` for examples.

## Troubleshooting

### Build Errors

**"error: failed to run custom build command for `tauri-build`"**
- Ensure Rust is installed: `rustc --version`
- Update Rust: `rustup update`

**"Cannot find module '@tauri-apps/api'"**
- Install dependencies: `pnpm install`

**Linux build fails with webkit2gtk errors**
- Install dependencies: `sudo apt install webkit2gtk-4.1-dev libayatana-appindicator3-dev`

### Runtime Issues

**App window is blank**
- Check dev server is running: http://localhost:5173
- Check browser console for errors
- Try `pnpm build` then `pnpm tauri:build` for production testing

**File picker doesn't work**
- Ensure `tauri-plugin-dialog` is initialized in `lib.rs`
- Check `tauri.conf.json` has dialog plugin configured
- Verify `useTauri.ts` is using correct import

## Resources

- [Tauri Documentation](https://tauri.app/)
- [Tauri API Reference](https://tauri.app/reference/javascript/api/)
- [Tauri Plugin Guide](https://tauri.app/plugin/)
- [Rust Tauri Crate Docs](https://docs.rs/tauri/)

## Future Enhancements

- [ ] System tray with recent files menu
- [ ] Deep linking (`fdtd://open?file=...`)
- [ ] Auto-update mechanism
- [ ] Drag-and-drop file support
- [ ] Recent files persistence
- [ ] File associations (`.h5` opens in app)
- [ ] Native notifications for long-running operations
