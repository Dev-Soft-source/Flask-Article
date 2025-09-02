#!/usr/bin/env python3
r"""
Enhanced Search Tool for the CopyscriptAI Project

This tool provides advanced code search capabilities for the CopyscriptAI project codebase.
It allows searching with context, showing matches with surrounding lines for better understanding.

Features:
- Search with regular expressions or literal strings
- Configurable number of context lines
- File extension filtering
- Directory inclusion/exclusion
- Special handling for prompt files
- Colored output for better readability
- Case-insensitive search option
- Files-only mode for quick overview

Usage examples:
  ./smart_search.py "function_name"                  # Basic search
  ./smart_search.py -r "def\\s+\\w+_paragraph"       # Regex search
  ./smart_search.py "PAA" -c 5                       # Show 5 context lines
  ./smart_search.py "error" -e py,md                 # Search .py and .md files
  ./smart_search.py "prompt" --include-dir script1   # Only search in script1 directory
  ./smart_search.py "function" -f                    # Show only matching files
  ./smart_search.py "paragraph" -p                   # Special prompt file handling
  ./smart_search.py -h                               # Show help

Author: CopyscriptAI Team
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Define colors for terminal output
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"

# Project base directory
BASE_DIR = Path("./")

# Default directories to exclude
DEFAULT_EXCLUDES = {
    "__pycache__",
    "venv",
    ".venv",
    "__MACOSX",
    ".git",
    "generated_articles",
    "article_contexts",
    "cache",
}

# Default file extensions to include
DEFAULT_EXTENSIONS = {".py", ".md", ".txt", ".json"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Search code with context in the CopyscriptAI project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  ./smart_search.py "function_name"                  # Basic search
  ./smart_search.py -r "def\\s+\\w+_paragraph"       # Regex search
  ./smart_search.py "PAA" -c 5                       # Show 5 context lines
  ./smart_search.py "error" -e py,md                 # Search .py and .md files
  ./smart_search.py "prompt" --include-dir script1   # Only search in script1 directory
  ./smart_search.py "grammar" -f                     # Show only matching files
        """
    )
    parser.add_argument(
        "pattern", type=str, help="Pattern to search for (supports basic regex by default)"
    )
    parser.add_argument(
        "-c", "--context", type=int, default=2, help="Number of context lines (default: 2)"
    )
    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        default=",".join(ext[1:] for ext in DEFAULT_EXTENSIONS),
        help="Comma-separated list of file extensions to search (without dots)",
    )
    parser.add_argument(
        "--include-dir", type=str, help="Only search in this directory (relative to project root)"
    )
    parser.add_argument(
        "--exclude-dir",
        type=str,
        help="Additional comma-separated directories to exclude",
    )
    parser.add_argument(
        "-i", "--case-insensitive", action="store_true", help="Case insensitive search"
    )
    parser.add_argument(
        "-f", "--files-only", action="store_true", help="Only show file names, not matches"
    )
    parser.add_argument(
        "-p", "--prompt-context", action="store_true", 
        help="Show extra context for LLM prompts (helps with LLM-related searches)"
    )
    parser.add_argument(
        "-r", "--regex", action="store_true", 
        help="Treat the pattern as a regular expression (more advanced than the basic regex support)"
    )
    parser.add_argument(
        "--no-color", action="store_true", 
        help="Disable colored output"
    )
    parser.add_argument(
        "--base-dir", type=str,
        help=f"Override the base directory (default: {BASE_DIR})"
    )
    return parser.parse_args()
    return parser.parse_args()


def should_exclude(path: Path, exclude_dirs: Set[str]) -> bool:
    """Check if a path should be excluded based on exclude directories."""
    path_parts = path.parts
    return any(exclude in path_parts for exclude in exclude_dirs)


def get_file_paths(
    base_dir: Path, 
    extensions: Set[str], 
    exclude_dirs: Set[str],
    include_dir: Optional[str] = None
) -> List[Path]:
    """Get all files with the specified extensions, excluding certain directories."""
    files = []
    search_dir = base_dir
    
    # If include_dir is specified, adjust the search directory
    if include_dir:
        search_dir = base_dir / include_dir
        if not search_dir.exists():
            print(f"Directory not found: {search_dir}")
            return []
    
    for root, dirs, filenames in os.walk(search_dir):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        root_path = Path(root)
        for filename in filenames:
            file_path = root_path / filename
            if file_path.suffix in extensions:
                files.append(file_path)
    
    return files


def highlight_match(line: str, pattern: str, case_insensitive: bool, is_regex: bool = False, colors: dict = None) -> str:
    """Highlight the matching pattern in the line."""
    if colors is None:
        colors = {"RED": RED, "RESET": RESET}
        
    flags = re.IGNORECASE if case_insensitive else 0
    
    # If not in regex mode and pattern isn't already escaped, escape it
    if not is_regex and not pattern.startswith('\\'):
        pattern = re.escape(pattern)
        
    return re.sub(
        pattern, 
        lambda m: f"{colors['RED']}{m.group(0)}{colors['RESET']}", 
        line, 
        flags=flags
    )


def search_file(
    file_path: Path, 
    pattern: str, 
    context_lines: int, 
    case_insensitive: bool,
    files_only: bool,
    prompt_context: bool,
    is_regex: bool = False
) -> List[Tuple[int, str, bool]]:
    """
    Search a file for the pattern and return matches with context.
    
    Returns a list of tuples (line_number, line_content, is_match)
    """
    flags = re.IGNORECASE if case_insensitive else 0
    results = []
    
    # If not in regex mode and pattern isn't already escaped, escape it
    if not is_regex and not pattern.startswith('\\'):
        search_pattern = re.escape(pattern)
    else:
        search_pattern = pattern
    
    # If we're only showing file names, just check if there's a match and return
    if files_only:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if re.search(search_pattern, content, flags):
                    return [(0, "", True)]  # Dummy match for files_only mode
        except UnicodeDecodeError:
            # Skip binary files
            pass
        return []
    
    # For full search with context
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Special handling for prompt files if requested
        if prompt_context and file_path.name == "prompts.py":
            return search_prompt_file(file_path, lines, search_pattern, flags)
        
        # Regular file search with context
        match_indices = []
        for i, line in enumerate(lines):
            if re.search(search_pattern, line, flags):
                match_indices.append(i)
        
        # Get context for each match
        seen_lines = set()
        for match_idx in match_indices:
            start = max(0, match_idx - context_lines)
            end = min(len(lines), match_idx + context_lines + 1)
            
            for i in range(start, end):
                if i not in seen_lines:
                    seen_lines.add(i)
                    is_match = i in match_indices
                    results.append((i + 1, lines[i].rstrip(), is_match))
            
            # Add a separator if not the last match
            if match_idx != match_indices[-1]:
                results.append((-1, "", False))
                
    except UnicodeDecodeError:
        # Skip binary files
        pass
    
    return results


def search_prompt_file(
    file_path: Path, 
    lines: List[str], 
    pattern: str, 
    flags: int
) -> List[Tuple[int, str, bool]]:
    """
    Special handling for prompts.py files to provide better context for LLM prompts.
    
    This looks for prompt definitions and includes the entire prompt context.
    """
    results = []
    in_prompt = False
    prompt_start = 0
    
    # First, identify complete prompt blocks
    prompt_blocks = []
    for i, line in enumerate(lines):
        # Look for patterns like "PARAGRAPH_GENERATION_PROMPT = """
        if re.search(r'[A-Z_]+ = [\'"]', line) or re.search(r'[A-Z_]+ = f?[\'"]', line) or re.search(r'[A-Z_]+ = """', line):
            if in_prompt:
                # End previous prompt block
                prompt_blocks.append((prompt_start, i))
            # Start new prompt block
            in_prompt = True
            prompt_start = i
        # Check for end of string marker at the start of a line
        elif (line.strip().startswith('"""') or line.strip().startswith("'''")) and in_prompt:
            in_prompt = False
            prompt_blocks.append((prompt_start, i + 1))
    
    # If still in a prompt at the end of the file
    if in_prompt:
        prompt_blocks.append((prompt_start, len(lines)))
    
    # Now search each prompt block for the pattern
    matches_found = False
    for start, end in prompt_blocks:
        # Check if the pattern exists in this block
        block_text = ''.join(lines[start:end])
        if not re.search(pattern, block_text, flags):
            continue
        
        matches_found = True
        
        # If there's a match, include the entire prompt block
        # First, add a separator if not the first block
        if results:
            results.append((-1, "", False))
        
        # Add a header with the prompt name
        prompt_name = re.search(r'([A-Z_]+) =', lines[start])
        if prompt_name:
            header = f"--- PROMPT: {prompt_name.group(1)} ---"
            results.append((-2, header, False))
        
        # Add all lines from this prompt block
        for i in range(start, end):
            is_match = bool(re.search(pattern, lines[i], flags))
            results.append((i + 1, lines[i].rstrip(), is_match))
    
    return results


def main():
    """Main function to search through the codebase."""
    args = parse_args()
    
    # Setup colors or disable them if requested
    colors = {
        "RESET": RESET,
        "RED": RED,
        "GREEN": GREEN,
        "YELLOW": YELLOW,
        "CYAN": CYAN
    }
    
    if args.no_color:
        colors = {k: "" for k in colors}
    
    # Use custom base directory if provided
    base_dir = Path(args.base_dir) if args.base_dir else BASE_DIR
    if not base_dir.exists():
        print(f"Error: Base directory '{base_dir}' not found")
        return 1
    
    # Prepare extensions
    extensions = {f".{ext}" for ext in args.extensions.split(",")} if args.extensions else DEFAULT_EXTENSIONS
    
    # Prepare excluded directories
    exclude_dirs = DEFAULT_EXCLUDES.copy()
    if args.exclude_dir:
        exclude_dirs.update(set(args.exclude_dir.split(",")))
    
    # Get files to search
    files = get_file_paths(base_dir, extensions, exclude_dirs, args.include_dir)
    
    # Track if we found any matches
    found_matches = False
    
    # Compile regex pattern if using regex mode
    if args.regex:
        try:
            pattern = args.pattern
            flags = re.IGNORECASE if args.case_insensitive else 0
            compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            print(f"Error in regular expression: {e}")
            return 1
    else:
        # In non-regex mode, escape special characters to treat as literal
        pattern = re.escape(args.pattern)
        # But we still use regex for search, just with escaped pattern
    
    # Display search parameters
    print(f"{colors['CYAN']}Searching for: '{args.pattern}'{colors['RESET']}")
    print(f"{colors['CYAN']}Mode: {'Regex' if args.regex else 'Literal string'}{colors['RESET']}")
    if args.include_dir:
        print(f"{colors['CYAN']}In directory: {args.include_dir}{colors['RESET']}")
    print(f"{colors['CYAN']}File types: {', '.join(ext[1:] for ext in extensions)}{colors['RESET']}")
    print()
    
    # Search each file
    for file_path in sorted(files):
        results = search_file(
            file_path, 
            pattern, 
            args.context, 
            args.case_insensitive,
            args.files_only,
            args.prompt_context,
            args.regex
        )
        
        if results:
            found_matches = True
            rel_path = file_path.relative_to(base_dir)
            
            # Print file header
            print(f"\n{colors['CYAN']}{rel_path}{colors['RESET']}")
            
            # For files-only mode, we just need to print the filename
            if args.files_only:
                continue
            
            # Print results with context
            for line_num, content, is_match in results:
                # Separator line
                if line_num == -1:
                    print("...")
                    continue
                
                # Special header line
                if line_num == -2:
                    print(f"{colors['YELLOW']}{content}{colors['RESET']}")
                    continue
                
                # Regular line with or without match
                if is_match:
                    prefix = f"{colors['GREEN']}{line_num:4d}{colors['RESET']}: "
                    highlighted = highlight_match(content, pattern, args.case_insensitive, args.regex, colors)
                    print(f"{prefix}{highlighted}")
                else:
                    prefix = f"{line_num:4d}: "
                    print(f"{prefix}{content}")
    
    # Show summary
    if not found_matches:
        print(f"\nNo matches found for '{args.pattern}'")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
