#!/bin/bash
# Setup Cloudflare Pages secrets for GitHub Actions deployment
#
# This script helps configure the required secrets for deploying
# strata-web to Cloudflare Pages via GitHub Actions.
#
# Prerequisites:
# - Cloudflare account with Pages enabled
# - GitHub CLI (gh) installed and authenticated
# - Repository admin access
# - Wrangler CLI (optional, for project creation)

set -e

echo "=== Cloudflare Pages Setup for strata-fdtd ==="
echo ""
echo "This script will help you configure GitHub secrets for"
echo "Cloudflare Pages deployment."
echo ""

# Check for gh CLI
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check gh auth status
if ! gh auth status &> /dev/null; then
    echo "Error: GitHub CLI is not authenticated."
    echo "Run: gh auth login"
    exit 1
fi

echo "=== Step 1: Get Cloudflare API Token ==="
echo ""
echo "1. Go to: https://dash.cloudflare.com/profile/api-tokens"
echo "2. Click 'Create Token'"
echo "3. Use template: 'Edit Cloudflare Workers'"
echo "4. Customize permissions:"
echo "   - Account > Cloudflare Pages > Edit"
echo "   - Account Resources > Include > Your Account"
echo "5. Create and copy the token (shown only once!)"
echo ""
read -p "Enter your Cloudflare API Token: " -s CLOUDFLARE_API_TOKEN
echo ""

if [ -z "$CLOUDFLARE_API_TOKEN" ]; then
    echo "Error: API token cannot be empty"
    exit 1
fi

echo ""
echo "=== Step 2: Get Cloudflare Account ID ==="
echo ""
echo "Find your Account ID:"
echo "- Go to: https://dash.cloudflare.com"
echo "- Look in the right sidebar under 'Account ID'"
echo "- Or check the URL: dash.cloudflare.com/{account-id}/..."
echo ""
read -p "Enter your Cloudflare Account ID: " CLOUDFLARE_ACCOUNT_ID

if [ -z "$CLOUDFLARE_ACCOUNT_ID" ]; then
    echo "Error: Account ID cannot be empty"
    exit 1
fi

echo ""
echo "=== Step 3: Configure GitHub Secrets ==="
echo ""

# Set secrets
echo "Setting CLOUDFLARE_API_TOKEN..."
echo "$CLOUDFLARE_API_TOKEN" | gh secret set CLOUDFLARE_API_TOKEN

echo "Setting CLOUDFLARE_ACCOUNT_ID..."
gh secret set CLOUDFLARE_ACCOUNT_ID --body "$CLOUDFLARE_ACCOUNT_ID"

echo ""
echo "=== Step 4: Create Cloudflare Pages Project ==="
echo ""

# Check if wrangler is available
if command -v wrangler &> /dev/null || command -v npx &> /dev/null; then
    read -p "Create Cloudflare Pages project 'strata-web' now? (y/N) " CREATE_PROJECT

    if [[ "$CREATE_PROJECT" =~ ^[Yy]$ ]]; then
        echo "Creating Cloudflare Pages project..."

        # Export credentials for wrangler
        export CLOUDFLARE_API_TOKEN
        export CLOUDFLARE_ACCOUNT_ID

        # Try to create the project (will fail gracefully if it exists)
        if command -v wrangler &> /dev/null; then
            wrangler pages project create strata-web --production-branch=main 2>/dev/null || echo "Project may already exist (this is OK)"
        else
            npx wrangler pages project create strata-web --production-branch=main 2>/dev/null || echo "Project may already exist (this is OK)"
        fi

        echo "Project created (or already exists)!"
    else
        echo "Skipping project creation."
        echo "The GitHub Actions workflow will create the project automatically on first deploy."
    fi
else
    echo "Note: Wrangler CLI not found. The GitHub Actions workflow will"
    echo "create the project automatically on the first deployment."
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "GitHub secrets configured:"
echo "  - CLOUDFLARE_API_TOKEN"
echo "  - CLOUDFLARE_ACCOUNT_ID"
echo ""
echo "Next steps:"
echo "1. Push a commit to main to trigger deployment"
echo "   git commit --allow-empty -m 'Trigger web deployment'"
echo "   git push"
echo ""
echo "2. Check deployment status in GitHub Actions"
echo ""
echo "Production URL: https://strata-web.pages.dev"
