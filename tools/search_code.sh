#!/bin/bash

###############################################################################
# search_code.sh - Advanced search script for the CopyscriptAI project
###############################################################################
#
# This script provides a convenient way to search through the project codebase
# while excluding common directories that aren't relevant to code analysis
# like __pycache__, virtual environments, and cache directories.
#
# Features:
# - Excludes irrelevant directories automatically
# - Supports regular expressions with -r flag
# - Configurable file types to search
# - Case-insensitive search option
# - Can limit search to specific directories
# - Colored output for better readability
#
# Usage:
#   ./search_code.sh [OPTIONS] "search pattern"
#
# Options:
#   -r, --regex         Use regular expression for search
#   -i, --ignore-case   Case insensitive search
#   -t, --type EXT      Specify file extensions to search (e.g., "py,md,txt")
#                       Default: py,md,txt,json
#   -d, --dir DIR       Only search in this directory (relative to project root)
#   -n, --no-color      Disable colored output
#   -h, --help          Show this help message
#
# Examples:
#   ./search_code.sh "function_name"                # Basic search
#   ./search_code.sh -r "def\s+\w+_paragraph"       # Regex search
#   ./search_code.sh -i "error"                     # Case-insensitive search
#   ./search_code.sh -t py,md "prompt"              # Search only .py and .md files
#   ./search_code.sh -d script1 "generate"          # Search only in script1 dir
#
# Author: CopyscriptAI Team
###############################################################################

# Default values
BASE_DIR="/home/abuh/Documents/Python/LLM_article_gen_2/scripts"
USE_REGEX=false
IGNORE_CASE=false
FILE_TYPES="py,md,txt,json"
SEARCH_DIR=""
USE_COLOR=true

# Function to display help
show_help() {
    grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--regex)
            USE_REGEX=true
            shift
            ;;
        -i|--ignore-case)
            IGNORE_CASE=true
            shift
            ;;
        -t|--type)
            FILE_TYPES="$2"
            shift 2
            ;;
        -d|--dir)
            SEARCH_DIR="$2"
            shift 2
            ;;
        -n|--no-color)
            USE_COLOR=false
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            # If the next argument is a flag or we've processed all flags
            if [[ "$1" == -* ]]; then
                echo "Unknown option: $1"
                show_help
            else
                PATTERN="$1"
                shift
                break
            fi
            ;;
    esac
done

# Check if pattern is provided
if [ -z "$PATTERN" ]; then
    echo "Error: No search pattern provided"
    show_help
fi

# Prepare grep options
GREP_OPTS="-r"
if [ "$USE_REGEX" = true ]; then
    GREP_OPTS="$GREP_OPTS -E"
fi
if [ "$IGNORE_CASE" = true ]; then
    GREP_OPTS="$GREP_OPTS -i"
fi
if [ "$USE_COLOR" = true ]; then
    GREP_OPTS="$GREP_OPTS --color=always"
fi

# Prepare file includes
IFS=',' read -ra EXTS <<< "$FILE_TYPES"
INCLUDE_OPTS=""
for ext in "${EXTS[@]}"; do
    INCLUDE_OPTS="$INCLUDE_OPTS --include=*.${ext}"
done

# Set search directory
if [ -n "$SEARCH_DIR" ]; then
    SEARCH_PATH="$BASE_DIR/$SEARCH_DIR"
    if [ ! -d "$SEARCH_PATH" ]; then
        echo "Error: Directory '$SEARCH_DIR' does not exist"
        exit 1
    fi
else
    SEARCH_PATH="$BASE_DIR"
fi

# Display search parameters
echo "Searching for: '$PATTERN'"
echo "Mode: $([ "$USE_REGEX" = true ] && echo "Regex" || echo "Literal string")"
echo "Path: $SEARCH_PATH"
echo "File types: $FILE_TYPES"
echo ""

# Build and execute the grep command
COMMAND="grep $GREP_OPTS $INCLUDE_OPTS \
    --exclude-dir=__pycache__ \
    --exclude-dir=venv \
    --exclude-dir=.venv \
    --exclude-dir=__MACOSX \
    --exclude-dir=.git \
    --exclude-dir=generated_articles \
    --exclude-dir=article_contexts \
    --exclude-dir=cache \
    \"$PATTERN\" \"$SEARCH_PATH\""

# Execute the command
eval $COMMAND

# Exit with grep's exit code
exit $?
