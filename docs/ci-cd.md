# CI/CD and Deployment Infrastructure

This document describes the continuous integration, deployment, and release infrastructure for the strata-fdtd project.

## Overview

The project uses GitHub Actions for CI/CD with workflows for:
- **Web UI**: Automated deployment to Vercel
- **Python CLI**: Multi-platform testing and coverage reporting
- **Desktop App**: Build and release automation

## Table of Contents

- [Secrets Configuration](#secrets-configuration)
- [Web UI Deployment](#web-ui-deployment)
- [Coverage Reporting](#coverage-reporting)
- [Desktop Releases](#desktop-releases)
- [Workflows](#workflows)
- [Troubleshooting](#troubleshooting)

## Secrets Configuration

All secrets are configured in **Settings → Secrets and variables → Actions** in the GitHub repository.

### Quick Setup

Use the provided helper scripts to configure secrets:

```bash
# Validate current configuration
scripts/validate-secrets.sh

# Configure Vercel deployment (required)
scripts/setup-vercel.sh

# Configure Codecov reporting (required)
scripts/setup-codecov.sh

# Configure macOS code signing (optional)
scripts/setup-apple-signing.sh

# Configure Tauri auto-update (optional)
scripts/setup-tauri-signing.sh
```

### Required Secrets

These secrets are required for CI/CD to function:

#### Web Deployment

| Secret | Description | How to Get |
|--------|-------------|------------|
| `VERCEL_TOKEN` | Vercel authentication token | https://vercel.com/account/tokens |
| `VERCEL_ORG_ID` | Vercel organization ID | Run `vercel link` in `web-ui/` |
| `VERCEL_PROJECT_ID` | Vercel project ID | Run `vercel link` in `web-ui/` |

#### Coverage Reporting

| Secret | Description | How to Get |
|--------|-------------|------------|
| `CODECOV_TOKEN` | Codecov upload token | https://codecov.io after linking repo |

### Optional Secrets

These secrets enable additional features but are not required:

#### Web Environment (Optional)

| Secret | Description | Example |
|--------|-------------|---------|
| `VITE_API_URL` | Production API endpoint | `https://api.example.com` |
| `DEPLOYED_URL` | Production URL for smoke tests | `https://fdtd-simulator.vercel.app` |

#### macOS Code Signing (Optional)

Required only for signed macOS releases:

| Secret | Description | How to Get |
|--------|-------------|------------|
| `APPLE_CERTIFICATE` | Base64-encoded .p12 certificate | Export from Keychain, encode with `base64` |
| `APPLE_CERTIFICATE_PASSWORD` | Certificate password | Set when exporting .p12 |
| `APPLE_SIGNING_IDENTITY` | Certificate identity | `security find-identity -v -p codesigning` |
| `APPLE_ID` | Apple ID for notarization | Your Apple Developer account email |
| `APPLE_PASSWORD` | App-specific password | https://appleid.apple.com |
| `APPLE_TEAM_ID` | Apple Developer Team ID | Apple Developer account settings |

#### Windows Code Signing (Optional)

Required only for signed Windows releases:

| Secret | Description | How to Get |
|--------|-------------|------------|
| `WINDOWS_CERTIFICATE` | Base64-encoded .pfx certificate | Export from certificate store, encode with `base64` |
| `WINDOWS_CERTIFICATE_PASSWORD` | Certificate password | Set when exporting .pfx |

#### Tauri Auto-Update (Optional)

Required only for auto-update functionality:

| Secret | Description | How to Get |
|--------|-------------|------------|
| `TAURI_SIGNING_PRIVATE_KEY` | Tauri updater signing key | `pnpm tauri signer generate` |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | Key password (if set) | Password used during key generation |

## Web UI Deployment

The Web UI is deployed automatically to Vercel on every push to the main branch.

### Setup

1. **Create Vercel Project**
   ```bash
   # Install Vercel CLI
   npm i -g vercel

   # Link project
   cd web-ui
   vercel link
   ```

2. **Configure Secrets**
   ```bash
   # Use helper script
   scripts/setup-vercel.sh

   # Or manually:
   gh secret set VERCEL_TOKEN
   gh secret set VERCEL_ORG_ID
   gh secret set VERCEL_PROJECT_ID
   ```

3. **Verify Deployment**
   - Push a commit to trigger workflow
   - Check Actions tab for deployment status
   - Visit deployed URL to verify

### Preview Deployments

Pull requests automatically get preview deployments:
- Each PR gets a unique preview URL
- Preview URL is commented on the PR
- Updates automatically on new commits

### Manual Deployment

Deploy manually if needed:
```bash
cd web-ui
vercel --prod  # Production deployment
vercel         # Preview deployment
```

## Coverage Reporting

Test coverage is automatically reported to Codecov for all CI runs.

### Setup

1. **Link Repository**
   - Go to https://codecov.io
   - Sign in with GitHub
   - Add repository: `rjwalters/strata-fdtd`

2. **Configure Token**
   ```bash
   # Use helper script
   scripts/setup-codecov.sh

   # Or manually:
   gh secret set CODECOV_TOKEN
   ```

3. **View Reports**
   - Visit https://app.codecov.io/gh/rjwalters/strata-fdtd
   - Reports update automatically on each CI run
   - Coverage changes shown on pull requests

### Coverage Targets

- **Minimum**: 80% overall coverage
- **New Code**: 90% coverage on new/modified code
- **Critical Paths**: 100% coverage on core functionality

## Desktop Releases

Desktop builds are created automatically on version tags using the `desktop-release.yml` workflow.

### Creating a Release

**Option 1: Use the Version Bump Workflow (Recommended)**

1. Go to **Actions → Version Bump** in GitHub
2. Click **Run workflow**
3. Enter the new version (e.g., `0.2.0`)
4. Check "Create and push tag" (default: true)
5. Click **Run workflow**

This automatically:
- Updates all package versions (package.json, Cargo.toml, tauri.conf.json, pyproject.toml)
- Commits the changes
- Creates and pushes the version tag
- Triggers the desktop release workflow

**Option 2: Manual Workflow**

```bash
# Update versions manually across all packages
# strata-app/package.json
# strata-web/package.json
# strata-ui/package.json
# strata-app/src-tauri/Cargo.toml
# strata-app/src-tauri/tauri.conf.json
# pyproject.toml

git add -u
git commit -m "chore: bump version to v1.2.3"
git tag v1.2.3
git push origin main --tags
```

### Release Process

1. Tag triggers `desktop-release.yml` workflow
2. GitHub Actions builds for all platforms in parallel
3. Builds are code-signed (if secrets are configured)
4. Draft release is created with all installers attached
5. Maintainer reviews and publishes the release

### Supported Platforms

| Platform | Architecture | Installer Format |
|----------|--------------|------------------|
| macOS | Universal (Intel + Apple Silicon) | `.dmg` |
| Linux | x86_64 | `.AppImage`, `.deb` |
| Windows | x86_64 | `.msi`, `.exe` |

### Signed Builds (macOS)

For signed and notarized macOS builds:

1. **Configure Apple Signing**
   ```bash
   scripts/setup-apple-signing.sh
   ```

2. **Requirements**
   - Valid Apple Developer account
   - Developer ID Application certificate
   - App-specific password for notarization

3. **Verification**
   ```bash
   # Check signature
   codesign -dv --verbose=4 YourApp.app

   # Check notarization
   spctl -a -vv YourApp.app
   ```

### Auto-Update (Tauri)

For Tauri auto-update functionality:

1. **Configure Update Signing**
   ```bash
   scripts/setup-tauri-signing.sh
   ```

2. **Add Public Key**
   - Public key automatically added to `tauri.conf.json`
   - Or manually add to `plugins.updater.pubkey`

3. **Update Configuration**
   ```json
   // web-ui/src-tauri/tauri.conf.json
   {
     "plugins": {
       "updater": {
         "active": true,
         "endpoints": [
           "https://github.com/rjwalters/ml-audio-codecs/releases/latest/download/latest.json"
         ],
         "pubkey": "YOUR_PUBLIC_KEY_HERE"
       }
     }
   }
   ```

## Workflows

### Web UI CI (`.github/workflows/web-ui-ci.yml`)

Triggers: Push to `main`, pull requests

Steps:
1. Install dependencies
2. Build web UI
3. Run type checking
4. Run tests
5. Deploy to Vercel (main branch only)
6. Smoke test deployment

### CLI CI (`.github/workflows/cli-ci.yml`)

Triggers: Push to `main`, pull requests

Steps:
1. Set up Python environment
2. Install dependencies
3. Run tests with coverage
4. Upload coverage to Codecov
5. Check code formatting (black, ruff)

Platforms:
- Ubuntu (latest)
- macOS (latest)
- Windows (latest)

### Desktop Release (`.github/workflows/desktop-release.yml`)

Triggers: Tags matching `v*.*.*`, manual dispatch

Steps:
1. Create draft GitHub release
2. Build for all platforms (macOS, Linux, Windows) in parallel
3. Sign builds (if code signing secrets are configured)
4. Notarize macOS builds (if Apple secrets are configured)
5. Upload installers to release
6. Generate auto-update manifest (if Tauri signing key is configured)

### Version Bump (`.github/workflows/version-bump.yml`)

Triggers: Manual dispatch only

Steps:
1. Validate version format (semver)
2. Update all package versions:
   - strata-app/package.json
   - strata-web/package.json
   - strata-ui/package.json
   - strata-app/src-tauri/Cargo.toml
   - strata-app/src-tauri/tauri.conf.json
   - pyproject.toml
3. Commit version changes
4. Create and push tag (optional)
5. Tag automatically triggers desktop-release workflow

## Troubleshooting

### Validation

Check secrets configuration:
```bash
scripts/validate-secrets.sh
```

### Common Issues

**Deployment Fails**
- Check Vercel token is valid
- Verify project is linked correctly
- Check build logs for errors

**Coverage Upload Fails**
- Verify Codecov token is set
- Check repository is linked on Codecov
- Ensure coverage files are generated

**macOS Signing Fails**
- Verify certificate is valid and not expired
- Check signing identity matches certificate
- Ensure app-specific password is correct
- Verify notarization status on Apple Developer portal

**Auto-Update Doesn't Work**
- Verify public key in tauri.conf.json
- Check private key secret is set
- Ensure endpoints URL is correct
- Verify release manifest is generated

### Debugging Workflows

Enable debug logging:
```bash
# Set in repository secrets
gh secret set ACTIONS_STEP_DEBUG --body "true"
gh secret set ACTIONS_RUNNER_DEBUG --body "true"
```

View workflow logs:
```bash
# List recent workflow runs
gh run list

# View specific run logs
gh run view <run-id> --log
```

### Manual Secret Management

```bash
# List all secrets
gh secret list

# Set a secret
gh secret set SECRET_NAME --body "value"

# Delete a secret
gh secret delete SECRET_NAME
```

## Security Best Practices

### Secret Management

- **Never commit secrets** to the repository
- **Rotate secrets regularly** (every 90 days recommended)
- **Use minimal permissions** for tokens
- **Limit secret access** to necessary workflows only
- **Audit secret usage** periodically

### Certificate Management

- **Store certificates securely** outside repository
- **Use hardware security** for signing keys when possible
- **Enable two-factor authentication** on Apple Developer account
- **Monitor certificate expiration** dates
- **Revoke compromised certificates** immediately

### Access Control

- **Require PR reviews** for workflow changes
- **Restrict who can push** to protected branches
- **Enable branch protection** rules
- **Audit workflow runs** regularly
- **Review Actions permissions** periodically

## References

- **GitHub Actions Documentation**: https://docs.github.com/en/actions
- **Vercel Documentation**: https://vercel.com/docs
- **Codecov Documentation**: https://docs.codecov.com
- **Tauri Documentation**: https://tauri.app/v1/guides/distribution/updater
- **Apple Developer Documentation**: https://developer.apple.com/documentation/security/notarizing_macos_software_before_distribution

## Support

For issues with CI/CD:
1. Check this documentation
2. Run `scripts/validate-secrets.sh`
3. Review workflow logs in Actions tab
4. Create an issue with relevant logs and error messages
