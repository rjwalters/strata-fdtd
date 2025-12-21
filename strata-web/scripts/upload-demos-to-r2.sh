#!/bin/bash
# Upload demo simulation data to Cloudflare R2
#
# Prerequisites:
#   - wrangler CLI installed and authenticated (wrangler login)
#   - R2 bucket created (run setup-r2.sh first)
#
# Usage:
#   ./scripts/upload-demos-to-r2.sh [demo-name]
#
# Examples:
#   ./scripts/upload-demos-to-r2.sh              # Upload all demos
#   ./scripts/upload-demos-to-r2.sh closed-pipe  # Upload specific demo

set -e

BUCKET_NAME="strata-demos"
DEMOS_DIR="../shared/demos"  # Relative to strata-web
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root for consistent paths
cd "$PROJECT_ROOT"
DEMOS_DIR="../shared/demos"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Strata Demo Upload to R2 ==="
echo ""

# Check wrangler auth
echo "Checking wrangler authentication..."
if ! wrangler whoami &> /dev/null; then
    echo -e "${RED}Error: Not logged in to Cloudflare. Run 'wrangler login' first.${NC}"
    exit 1
fi
echo -e "${GREEN}Authenticated.${NC}"
echo ""

# Check if demos directory exists
if [ ! -d "$DEMOS_DIR" ]; then
    echo -e "${RED}Error: Demos directory not found at $DEMOS_DIR${NC}"
    exit 1
fi

# Function to upload a single demo
upload_demo() {
    local demo_path="$1"
    local demo_name="$(basename "$demo_path")"
    local parent_dir="$(basename "$(dirname "$demo_path")")"
    local r2_prefix="demos/$parent_dir/$demo_name"

    echo ""
    echo -e "${YELLOW}Uploading: $demo_name${NC}"
    echo "  Source: $demo_path"
    echo "  Destination: $BUCKET_NAME/$r2_prefix/"

    # Count files
    local file_count=$(find "$demo_path" -type f | wc -l | tr -d ' ')
    echo "  Files: $file_count"

    # Calculate total size
    local total_size=$(du -sh "$demo_path" | cut -f1)
    echo "  Size: $total_size"
    echo ""

    # Upload each file
    local uploaded=0
    local failed=0

    find "$demo_path" -type f | while read -r file; do
        local rel_path="${file#$demo_path/}"
        local r2_key="$r2_prefix/$rel_path"

        # Determine content type
        local content_type="application/octet-stream"
        case "$file" in
            *.json) content_type="application/json" ;;
            *.bin) content_type="application/octet-stream" ;;
            *.h5|*.hdf5) content_type="application/x-hdf5" ;;
        esac

        echo -n "  Uploading $rel_path... "
        if wrangler r2 object put "$BUCKET_NAME/$r2_key" \
            --file="$file" \
            --content-type="$content_type" \
            --remote \
            2>/dev/null; then
            echo -e "${GREEN}OK${NC}"
            ((uploaded++)) || true
        else
            echo -e "${RED}FAILED${NC}"
            ((failed++)) || true
        fi
    done

    echo ""
    echo -e "${GREEN}Demo '$demo_name' upload complete.${NC}"
}

# Determine which demos to upload
SPECIFIC_DEMO="$1"

if [ -n "$SPECIFIC_DEMO" ]; then
    # Upload specific demo
    # Find the demo directory
    DEMO_PATH=$(find "$DEMOS_DIR" -type d -name "$SPECIFIC_DEMO" | head -1)

    if [ -z "$DEMO_PATH" ]; then
        echo -e "${RED}Error: Demo '$SPECIFIC_DEMO' not found in $DEMOS_DIR${NC}"
        exit 1
    fi

    upload_demo "$DEMO_PATH"
else
    # Upload all demos
    echo "Discovering demos..."

    # Find all demo directories (directories containing *_manifest.json)
    find "$DEMOS_DIR" -name "*_manifest.json" -type f | while read -r manifest; do
        demo_dir="$(dirname "$manifest")"
        upload_demo "$demo_dir"
    done
fi

echo ""
echo "=== Upload Complete ==="
echo ""
echo "Files are now available at your R2 public URL."
echo "Update VITE_R2_BUCKET_URL in your .env file if needed."
