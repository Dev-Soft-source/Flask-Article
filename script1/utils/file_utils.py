# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import os
import sys
from typing import List, Optional, Tuple
from pathlib import Path

# Add parent directory to path to import unified_csv_processor
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # script1 directory
root_dir = os.path.dirname(parent_dir)  # project root directory

# Make sure the root directory is in the path
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import the UnifiedCSVProcessor from the common utils folder
# Use a direct import with the full path to avoid module confusion
import importlib.util
spec = importlib.util.spec_from_file_location(
    "unified_csv_processor",
    os.path.join(root_dir, "utils", "unified_csv_processor.py")
)
unified_csv_processor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unified_csv_processor_module)
UnifiedCSVProcessor = unified_csv_processor_module.UnifiedCSVProcessor

def validate_and_extract_lines(file_path: str) -> Optional[List[str]]:
    """
    Validate and extract lines from a text file.

    Args:
        file_path (str): Path to the text file
    Returns:
        Optional[List[str]]: List of extracted lines or None if validation fails
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.rule import Rule

    console = Console()

    # Create a colorful header
    console.rule("[bold rainbow]START TXT VALIDATION[/]", align="center")

    # Check if file exists
    if not os.path.exists(file_path):
        console.print(Panel(
            "[bold red]File not found![/]",
            title="[bold red]ERROR[/]",
            border_style="red"
        ))
        console.rule("[bold rainbow]DONE TXT VALIDATION[/]", align="center")
        return None

    console.print(Panel(
        f"[bold green]File found:[/] [blue]{file_path}[/]",
        title="[bold green]SUCCESS[/]",
        border_style="green"
    ))

    # Open and validate TXT
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        console.print(Panel(
            f"[bold red]Failed to read file:[/] {str(e)}",
            title="[bold red]ERROR[/]",
            border_style="red"
        ))
        return None

    # Check if file is empty
    if not lines:
        console.print(Panel(
            "[bold red]File is empty![/]",
            title="[bold red]ERROR[/]",
            border_style="red"
        ))
        console.rule("[bold rainbow]DONE TXT VALIDATION[/]", align="center")
        return None

    # Display file information
    console.print(f"[bold purple]Header detected:[/] [cyan]['Article Title'][/]")
    console.print(f"[bold purple]Number of columns:[/] [cyan]1[/]")
    console.print(f"[bold green]Extracting from column:[/] [bright_cyan]Article Title[/]")

    console.print(Panel(
        f"[bold green]Total articles found:[/] [bright_cyan]{len(lines)}[/]",
        title="[bold green]SUCCESS[/]",
        border_style="green"
    ))

    # Create a table for the extracted articles
    table = Table(
        title="[bold magenta]Extracted Articles[/]",
        show_header=True,
        header_style="bold bright_magenta",
        border_style="bright_magenta"
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Article Title", style="bright_cyan")

    # Add rows to the table
    for idx, line in enumerate(lines, start=1):
        table.add_row(str(idx), line)

    # Print the table
    console.print(table)

    # Create a colorful footer
    console.rule("[bold rainbow]DONE TXT VALIDATION[/]", align="center")

    return lines

def read_keywords_file(file_path: str) -> List[Tuple[str, str]]:
    """
    Read keywords file and return list of (keyword, image_keyword) tuples.

    This function uses the UnifiedCSVProcessor to handle both simple keyword lists
    and structured CSV files with improved header detection and case-insensitive matching.

    Args:
        file_path (str): Path to the keywords file
    Returns:
        List[Tuple[str, str]]: List of (keyword, image_keyword) tuples
    """
    try:
        # Initialize the unified CSV processor
        processor = UnifiedCSVProcessor(file_path)

        # Use the validate_file method to display colorful validation output
        is_valid, message = processor.validate_file()

        if not is_valid:
            return []

        # Process the file
        result = processor.process_file()

        # Extract keywords from the result
        if result:
            if isinstance(result, list):
                # Simple mode - result is already a list of (keyword, image_keyword) tuples
                keywords = result
            else:
                # Structured mode - extract keywords from dictionary
                keywords = []
                for article_data in result.values():
                    keyword = article_data.get('keyword', '')
                    image_keyword = article_data.get('featured_img', keyword)
                    keywords.append((keyword, image_keyword))

            # Display the structure with colorful formatting
            processor.display_csv_structure()

            return keywords
        else:
            return []

    except Exception as e:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        console.print(Panel(
            f"[bold red]Error reading keywords file:[/] {str(e)}",
            title="[bold red]ERROR[/]",
            border_style="red"
        ))
        return []

def ensure_directory_exists(directory: str) -> bool:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        directory (str): Directory path
    Returns:
        bool: True if directory exists or was created, False otherwise
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            console.print(f"[green]Created directory:[/] [cyan]{directory}[/]")
        return True
    except Exception as e:
        console.print(f"[red]Failed to create directory {directory}:[/] [yellow]{str(e)}[/]")
        return False

def clean_filename(filename: str) -> str:
    """
    Clean a filename by removing invalid characters.

    Args:
        filename (str): Original filename
    Returns:
        str: Cleaned filename
    """
    # Replace spaces with hyphens
    filename = filename.replace(' ', '-')

    # Remove invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '')

    return filename.lower()