#!/bin/bash
# Setup Cloudflare R2 bucket for Strata demo data
#
# Prerequisites:
#   - wrangler CLI installed and authenticated (wrangler login)
#   - Cloudflare account with R2 enabled
#
# Usage:
#   ./scripts/setup-r2.sh

set -e

BUCKET_NAME="strata-demos"

echo "=== Strata R2 Bucket Setup ==="
echo ""

# Check wrangler auth
echo "Checking wrangler authentication..."
if ! wrangler whoami &> /dev/null; then
    echo "Error: Not logged in to Cloudflare. Run 'wrangler login' first."
    exit 1
fi

ACCOUNT_ID=$(wrangler whoami 2>/dev/null | grep -o 'account_id = "[^"]*"' | cut -d'"' -f2 || true)
echo "Account: $(wrangler whoami 2>&1 | head -1)"
echo ""

# Create the bucket
echo "Creating R2 bucket: $BUCKET_NAME"
if wrangler r2 bucket create "$BUCKET_NAME" 2>&1 | grep -q "already exists"; then
    echo "Bucket '$BUCKET_NAME' already exists."
else
    echo "Bucket '$BUCKET_NAME' created successfully."
fi
echo ""

# Configure CORS for web access
echo "Configuring CORS policy..."
CORS_CONFIG=$(cat <<'EOF'
[
  {
    "AllowedOrigins": ["*"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["*"],
    "MaxAgeSeconds": 86400
  }
]
EOF
)

# Note: CORS configuration via wrangler requires using the API
# For now, output instructions for manual configuration
echo ""
echo "=== Manual Steps Required ==="
echo ""
echo "1. Enable public access for the bucket:"
echo "   - Go to Cloudflare Dashboard > R2 > $BUCKET_NAME > Settings"
echo "   - Under 'Public access', click 'Allow Access'"
echo "   - Choose 'R2.dev subdomain' for the public URL"
echo ""
echo "2. Configure CORS (if needed for custom domain):"
echo "   The CORS policy should allow GET requests from any origin."
echo ""
echo "3. Note your public bucket URL:"
echo "   It will be in the format: https://pub-<id>.r2.dev"
echo "   Or use a custom domain via Cloudflare."
echo ""
echo "4. Update the .env file with your bucket URL:"
echo "   VITE_R2_BUCKET_URL=https://pub-<id>.r2.dev"
echo ""
echo "=== Setup Complete ==="
