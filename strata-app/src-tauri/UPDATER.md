# Auto-Update Configuration

This document describes how to configure the Tauri auto-update mechanism for production releases.

## Security Requirements

**IMPORTANT:** Before releasing to production, you MUST configure a public key for update verification.

The `pubkey` field in `tauri.conf.json` is currently empty. Without a valid public key, updates will not be cryptographically verified, making the application vulnerable to man-in-the-middle attacks.

## Setup Instructions

### 1. Generate a Key Pair

Use the Tauri CLI to generate a signing key pair:

```bash
pnpm tauri signer generate -w ~/.tauri/fdtd-simulator.key
```

This creates:
- A private key at `~/.tauri/fdtd-simulator.key` (keep this secret!)
- A public key displayed in the terminal

### 2. Configure the Public Key

Add the public key to `tauri.conf.json`:

```json
{
  "plugins": {
    "updater": {
      "endpoints": [
        "https://github.com/rjwalters/strata-fdtd/releases/latest/download/latest.json"
      ],
      "pubkey": "dW50cnVzdGVkIGNvbW1lbnQ6... YOUR_PUBLIC_KEY_HERE"
    }
  }
}
```

### 3. Set Environment Variable for Signing

When building releases, set the private key path:

```bash
export TAURI_SIGNING_PRIVATE_KEY=$(cat ~/.tauri/fdtd-simulator.key)
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD=""  # If you set a password
```

### 4. Create Update Manifest

Each release needs a `latest.json` file uploaded to the release assets. The Tauri build process generates this automatically when the signing key is configured.

Example `latest.json` structure:
```json
{
  "version": "1.0.0",
  "notes": "Release notes here",
  "pub_date": "2024-01-01T00:00:00Z",
  "platforms": {
    "darwin-x86_64": {
      "signature": "...",
      "url": "https://github.com/.../releases/download/v1.0.0/app.app.tar.gz"
    },
    "darwin-aarch64": {
      "signature": "...",
      "url": "https://github.com/.../releases/download/v1.0.0/app.app.tar.gz"
    }
  }
}
```

## CI/CD Integration

For GitHub Actions, store the private key as a secret:

1. Go to repository Settings > Secrets and variables > Actions
2. Add `TAURI_SIGNING_PRIVATE_KEY` with your private key content
3. Add `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` if you set a password

Example workflow step:
```yaml
- name: Build Tauri App
  env:
    TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}
    TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY_PASSWORD }}
  run: pnpm tauri build
```

## References

- [Tauri Updater Documentation](https://tauri.app/v1/guides/distribution/updater)
- [Tauri Code Signing](https://tauri.app/v1/guides/distribution/sign-macos)
