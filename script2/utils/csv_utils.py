# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from config import Config
from utils.rich_provider import provider
from rich.table import Table

# Add parent directory to path to import unified_csv_processor
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # script2 directory
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

class CSVProcessor:
    """
    Handles CSV file processing for article generation with improved validation and error handling.

    This class now uses the UnifiedCSVProcessor internally to provide better compatibility
    with different CSV formats, improved header detection, and case-insensitive matching.
    """

    def __init__(self, file_path: str, config: Config):
        """
        Initialize the CSV processor.

        Args:
            file_path (str): Path to the CSV file
            config (Config): Configuration object
        """
        self.file_path = Path(file_path)
        self.config = config
        self.data = {}
        self.normalized_headers = {}  # Maps normalized header names to actual header names

        # Initialize the unified CSV processor with flexible parsing enabled
        self.unified_processor = UnifiedCSVProcessor(file_path, config)
        
        # Ensure flexible parsing is enabled if config has it set
        if hasattr(config, 'CSV_FLEXIBLE_PARSING'):
            self.unified_processor.enable_flexible_parsing = config.CSV_FLEXIBLE_PARSING
        if hasattr(config, 'CSV_MAX_SUBTITLES'):
            self.unified_processor.max_subtitles = config.CSV_MAX_SUBTITLES
        if hasattr(config, 'CSV_SUBTITLE_PATTERNS'):
            self.unified_processor.subtitle_patterns = config.CSV_SUBTITLE_PATTERNS
        if hasattr(config, 'CSV_IMAGE_PATTERNS'):
            self.unified_processor.image_patterns = config.CSV_IMAGE_PATTERNS

    def validate_file(self) -> Tuple[bool, str]:
        """
        Validates if the CSV file exists and has the correct format.

        This method uses the UnifiedCSVProcessor for validation, which provides
        improved header detection and more flexible column mapping.

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Use the unified processor for validation
        is_valid, message = self.unified_processor.validate_file()

        # If validation succeeded, copy the normalized headers
        if is_valid:
            self.normalized_headers = self.unified_processor.normalized_headers

        return is_valid, message

    def display_csv_template(self) -> None:
        """Displays a template for the expected CSV format."""
        # Use the unified processor to display the template
        self.unified_processor.display_csv_template()

    def process_file(self) -> Dict[int, Dict[str, Any]]:
        """
        Processes the CSV file and returns structured data.

        This method uses the UnifiedCSVProcessor for processing, which provides
        improved header detection, case-insensitive matching, and more flexible column mapping.

        Returns:
            Dict[int, Dict[str, Any]]: Dictionary of article data indexed by row number
        """
        # Use the unified processor for processing only - skip validation display
        result = self.unified_processor.process_file()

        try:
            # If the result is a list of tuples (simple mode), convert it to the expected format
            if isinstance(result, list):
                # Convert list of (keyword, image_keyword) tuples to dictionary format
                for index, (keyword, image_keyword) in enumerate(result, start=1):
                    self.data[index] = {
                        'keyword': keyword,
                        'featured_img': image_keyword
                    }
            else:
                # Otherwise, use the structured data directly
                self.data = result
                
                # Process the new subtitle_image_pairs structure if present
                for index, article_data in self.data.items():
                    if "subtitle_image_pairs" in article_data:
                        # Get the subtitle/image pairs
                        pairs = article_data.pop("subtitle_image_pairs")
                        
                        # Make sure we have the pairs in the traditional format for backward compatibility
                        for i, pair in enumerate(pairs, 1):
                            if f"subtitle{i}" not in article_data and pair["subtitle"]:
                                article_data[f"subtitle{i}"] = pair["subtitle"]
                                article_data[f"img{i}"] = pair["image"] if pair["image"] else ""

            return self.data

        except Exception as e:
            provider.error(f"Error processing CSV: {str(e)}")
            import traceback
            provider.error(f"Detailed error: {traceback.format_exc()}")
            return {}

    def get_article_data(self, index: int) -> Optional[Dict[str, Any]]:
        """Retrieves article data for a specific index."""
        return self.data.get(index)

    def get_total_articles(self) -> int:
        """Returns the total number of articles in the CSV."""
        return len(self.data)

    def get_all_keywords(self) -> List[str]:
        """Returns a list of all keywords from the CSV."""
        return [data.get('keyword', '') for data in self.data.values()]

    def display_csv_structure(self) -> None:
        """Displays the structure of the processed CSV file."""
        if not self.data:
            provider.warning("No CSV data to display. Process a file first.")
            return

        # Use the unified processor to display the structure
        # This will use the colorful output from the UnifiedCSVProcessor
        self.unified_processor.display_csv_structure()

    def validate_and_process(self) -> Tuple[bool, str]:
        """
        Validates and processes the CSV file in one step.

        This method uses the UnifiedCSVProcessor for validation and processing,
        which provides improved header detection and more flexible column mapping.

        Returns:
            Tuple[bool, str]: (success, message)
        """
        # First validate the file
        is_valid, message = self.validate_file()
        if not is_valid:
            provider.error(message)
            self.display_csv_template()
            return False, message

        # If valid, process the file
        self.data = self.process_file()
        if not self.data:
            provider.error("Failed to process CSV file")
            return False, "Failed to process CSV file"

        # Create and display a single summary table
        provider.info("────────────────────────── ARTICLE SUBTITLE COUNTS ──────────────────────────")
        table = Table(title="Article Subtitle Counts", show_header=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Article Keyword", style="cyan", width=30)
        table.add_column("Number of Subtitles", style="green", justify="center")

        for index, article_data in self.data.items():
            subtitle_count = sum(1 for key in article_data if key.startswith("subtitle") and article_data[key].strip())
            keyword = article_data.get('keyword', 'Unknown')
            table.add_row(str(index), keyword, str(subtitle_count))

        provider.console.print(table)
        provider.info("────────────────────────── END OF SUBTITLE COUNTS ──────────────────────────")

        return True, f"Successfully processed {len(self.data)} articles"