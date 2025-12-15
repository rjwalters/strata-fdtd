#!/bin/bash
# read-debug-logs.sh - Read the most recent debug logs from strata-app
#
# Usage:
#   ./read-debug-logs.sh           # Show most recent log file
#   ./read-debug-logs.sh --errors  # Show only errors
#   ./read-debug-logs.sh --tail 50 # Show last 50 lines
#   ./read-debug-logs.sh --list    # List all log files

DESKTOP_DIR="$HOME/Desktop"
LOG_PATTERN="strata-debug-*.txt"

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Read debug logs from strata-app"
    echo ""
    echo "Options:"
    echo "  --errors     Show only error lines"
    echo "  --warnings   Show only warning lines"
    echo "  --tail N     Show last N lines (default: all)"
    echo "  --list       List all available log files"
    echo "  --latest     Show the most recent log file (default)"
    echo "  --all        Concatenate all log files"
    echo "  --help       Show this help message"
}

list_logs() {
    echo "Available debug logs:"
    ls -lt "$DESKTOP_DIR"/$LOG_PATTERN 2>/dev/null || echo "No debug logs found on Desktop"
}

get_latest_log() {
    ls -t "$DESKTOP_DIR"/$LOG_PATTERN 2>/dev/null | head -1
}

# Parse arguments
FILTER=""
TAIL_LINES=""
MODE="latest"

while [[ $# -gt 0 ]]; do
    case $1 in
        --errors)
            FILTER="ERROR"
            shift
            ;;
        --warnings)
            FILTER="WARN"
            shift
            ;;
        --tail)
            TAIL_LINES="$2"
            shift 2
            ;;
        --list)
            MODE="list"
            shift
            ;;
        --latest)
            MODE="latest"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute based on mode
case $MODE in
    list)
        list_logs
        ;;
    latest)
        LOG_FILE=$(get_latest_log)
        if [[ -z "$LOG_FILE" ]]; then
            echo "No debug logs found on Desktop"
            echo ""
            echo "To generate logs:"
            echo "1. Open the Tauri app"
            echo "2. Trigger an error, or run in DevTools console:"
            echo "   debugLogger.writeLogsToFile()"
            exit 1
        fi

        echo "=== Reading: $LOG_FILE ==="
        echo ""

        if [[ -n "$FILTER" ]]; then
            if [[ -n "$TAIL_LINES" ]]; then
                grep "\[$FILTER\]" "$LOG_FILE" | tail -n "$TAIL_LINES"
            else
                grep "\[$FILTER\]" "$LOG_FILE"
            fi
        elif [[ -n "$TAIL_LINES" ]]; then
            tail -n "$TAIL_LINES" "$LOG_FILE"
        else
            cat "$LOG_FILE"
        fi
        ;;
    all)
        for f in $(ls -t "$DESKTOP_DIR"/$LOG_PATTERN 2>/dev/null); do
            echo "=== $f ==="
            if [[ -n "$FILTER" ]]; then
                grep "\[$FILTER\]" "$f"
            else
                cat "$f"
            fi
            echo ""
        done
        ;;
esac
