# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import csv
import os
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.rule import Rule

# Initialize console for rich output
console = Console()

class UnifiedCSVProcessor:
    """
    Unified CSV processor that handles both simple keyword lists and structured CSV files.

    This class provides backward compatibility with both script1 and script2 CSV formats,
    with improved header detection, case-insensitive matching, and flexible column mapping.
    """

    def __init__(self, file_path: str, config=None):
        """
        Initialize the CSV processor.

        Args:
            file_path (str): Path to the CSV file
            config (Optional): Configuration object (required for structured mode)
        """
        self.file_path = Path(file_path)
        self.config = config
        self.data = {}
        self.normalized_headers = {}  # Maps normalized header names to actual header names
        self.simple_mode = False  # Whether to use simple mode (script1 style)
        self.has_header = None  # Whether the file has a header row
        
        # Dynamic column detection properties
        self.dynamic_subtitle_columns = []  # Dynamically detected subtitle columns
        self.dynamic_image_columns = []  # Dynamically detected image columns
        self.enable_flexible_parsing = True  # Whether to enable flexible column parsing
        self.max_subtitles = 20  # Maximum number of subtitles to detect
        self.subtitle_patterns = ['subtitle', 'sub', 'heading', 'section']  # Patterns for subtitle columns
        self.image_patterns = ['img', 'image', 'pic', 'photo']  # Patterns for image columns

        # Default required columns if config is not provided
        self.required_columns = ["keyword"]
        self.optional_columns = ["featured_img"]
        self.subtitle_columns = []
        self.image_columns = []

        # If config is provided, use its column definitions
        if config:
            if hasattr(config, 'csv_required_columns'):
                self.required_columns = config.csv_required_columns
            if hasattr(config, 'csv_optional_columns'):
                self.optional_columns = config.csv_optional_columns
            if hasattr(config, 'csv_subtitle_columns'):
                self.subtitle_columns = config.csv_subtitle_columns
            if hasattr(config, 'csv_image_columns'):
                self.image_columns = config.csv_image_columns
                
            # Load flexible parsing configuration if available
            if hasattr(config, 'CSV_FLEXIBLE_PARSING'):
                self.enable_flexible_parsing = config.CSV_FLEXIBLE_PARSING
            if hasattr(config, 'CSV_MAX_SUBTITLES'):
                self.max_subtitles = config.CSV_MAX_SUBTITLES
            if hasattr(config, 'CSV_SUBTITLE_PATTERNS'):
                self.subtitle_patterns = config.CSV_SUBTITLE_PATTERNS
            if hasattr(config, 'CSV_IMAGE_PATTERNS'):
                self.image_patterns = config.CSV_IMAGE_PATTERNS

    def detect_file_type(self) -> bool:
        """
        Detect the file type and set the appropriate mode.

        Returns:
            bool: True if detection was successful, False otherwise
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # Read the first few lines to analyze
                lines = [line.strip() for line in f if line.strip()][:5]

                if not lines:
                    console.print("[red]Content file not found or empty[/]")
                    return False

                # Check if this is likely a simple keyword list
                # Simple keyword lists typically have 1-2 columns with no complex headers
                first_line = lines[0]
                parts = first_line.split(',')

                # For keywords with common words like "cat", "how to", etc.,
                # we want to treat the first line as data, not a header
                first_line_lower = first_line.lower()
                common_keyword_phrases = ['how to', 'what is', 'why ', 'when ', 'where ', 'which ', 'who ', 'can ', 'should ']
                common_keyword_words = ['cat', 'dog', 'vs', 'best', 'top', 'guide', 'tips', 'ways', 'review']

                # Check if the first line contains common keyword phrases or words
                contains_keyword_phrase = any(phrase in first_line_lower for phrase in common_keyword_phrases)
                contains_keyword_word = any(word in first_line_lower.split() for word in common_keyword_words)

                # If it contains common keyword phrases or words, it's likely a keyword, not a header
                if contains_keyword_phrase or contains_keyword_word:
                    self.has_header = False
                else:
                    # Auto-detect header using the standard method
                    self.has_header = self._detect_header(first_line)

                # If there are only 1-2 columns and no complex structure, use simple mode
                if len(parts) <= 2 and not self._is_structured_csv(lines):
                    self.simple_mode = True
                else:
                    self.simple_mode = False

                console.print(f"Content file found: {self.file_path}")
                return True
        except Exception as e:
            console.print(f"[red]Error detecting file type:[/] {str(e)}")
            return False

    def _detect_header(self, first_line: str) -> bool:
        """
        Detect if the first line is a header.

        Args:
            first_line (str): The first line of the file

        Returns:
            bool: True if the first line appears to be a header, False otherwise
        """
        first_line_lower = first_line.lower()

        # Check for exact header keywords (with word boundaries)
        header_indicators = ['keyword', 'title', 'featured', 'image', 'subtitle', 'img']

        # Split the line into parts by comma
        parts = first_line_lower.split(',')

        # Check if any part exactly matches a header indicator (exact match only)
        # This is more strict to avoid false positives
        for part in parts:
            part = part.strip()
            if part in header_indicators:
                return True

        # If the line contains common question words or phrases, it's likely a keyword, not a header
        common_keyword_phrases = ['how to', 'what is', 'why ', 'when ', 'where ', 'which ', 'who ', 'can ', 'should ']
        for phrase in common_keyword_phrases:
            if phrase in first_line_lower:
                return False

        # If the line contains common keyword words, it's likely a keyword, not a header
        common_keyword_words = ['cat', 'dog', 'vs', 'best', 'top', 'guide', 'tips', 'ways', 'review']
        for word in common_keyword_words:
            if word in first_line_lower.split():
                return False

        # If the line has multiple words with spaces, it's likely a keyword, not a header
        if ' ' in first_line and len(first_line.split()) > 2:
            return False

        # Only consider it a header if it's a very simple, short word or phrase
        # that matches typical header patterns
        return len(first_line.split()) <= 2 and len(first_line) < 15

    def _is_structured_csv(self, lines: List[str]) -> bool:
        """
        Determine if the CSV appears to have a structured format with subtitle/image pairs.

        Args:
            lines (List[str]): Sample lines from the CSV

        Returns:
            bool: True if the CSV appears to have a structured format, False otherwise
        """
        if not lines:
            return False

        # Check if the first line contains subtitle/image pair indicators
        first_line_lower = lines[0].lower()

        # Look for subtitle/image patterns with enhanced detection
        # Check for core required columns
        for col in self.required_columns:
            if col.lower() in first_line_lower:
                # If we find a required column, look for subtitle patterns
                for pattern in self.subtitle_patterns:
                    # Look for numbered patterns like subtitle1, sub2, etc.
                    numbered_pattern = re.compile(f'{pattern}\\d+', re.IGNORECASE)
                    if numbered_pattern.search(first_line_lower):
                        return True
                
                # Look for image patterns
                for pattern in self.image_patterns:
                    # Look for numbered patterns like img1, image2, etc.
                    numbered_pattern = re.compile(f'{pattern}\\d+', re.IGNORECASE)
                    if numbered_pattern.search(first_line_lower):
                        return True

        # Fall back to legacy detection patterns
        subtitle_pattern = re.compile(r'subtitle\d+|section\d+|sub\d+|heading\d+', re.IGNORECASE)
        image_pattern = re.compile(r'img\d+|image\d+|pic\d+|photo\d+', re.IGNORECASE)

        return bool(subtitle_pattern.search(first_line_lower) or image_pattern.search(first_line_lower))

    def process_file(self) -> Union[List[Tuple[str, str]], Dict[int, Dict[str, Any]]]:
        """
        Process the CSV file in the appropriate mode.

        Returns:
            Union[List[Tuple[str, str]], Dict[int, Dict[str, Any]]]:
                Either a list of (keyword, image_keyword) tuples (simple mode)
                or a dictionary of article data indexed by row number (structured mode)
        """
        # Detect file type if not already done
        if self.has_header is None:
            self.detect_file_type()

        if self.simple_mode:
            return self.process_simple_file()
        else:
            return self.process_structured_file()

    def process_simple_file(self) -> List[Tuple[str, str]]:
        """
        Process a simple keyword list file (script1 style).

        Returns:
            List[Tuple[str, str]]: List of (keyword, image_keyword) tuples
        """
        keywords = []

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

                # Skip header if present
                start_index = 1 if self.has_header else 0

                for line in lines[start_index:]:
                    parts = line.split(',')

                    if len(parts) >= 2:
                        keyword = parts[0].strip()
                        image_keyword = parts[1].strip()
                    else:
                        keyword = parts[0].strip()
                        image_keyword = keyword  # Use keyword as image keyword if not specified

                    keywords.append((keyword, image_keyword))

            # Store the data for later use
            self.data = keywords

            return keywords

        except Exception as e:
            console.print(f"[red]Error processing simple file:[/] {str(e)}")
            return []

    def process_structured_file(self) -> Dict[int, Dict[str, Any]]:
        """
        Process a structured CSV file (script2 style).

        Returns:
            Dict[int, Dict[str, Any]]: Dictionary of article data indexed by row number
        """
        # Validate file first
        is_valid, message = self.validate_file()

        if not is_valid:
            console.print(f"[red]Validation failed:[/] {message}")
            return {}

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)  # Get the header row
                
                # Process each row
                for index, row in enumerate(reader, start=1):
                    # Create a dictionary mapping header names to values, handling short rows
                    row_dict = {}
                    for i, header in enumerate(headers):
                        row_dict[header] = row[i].strip() if i < len(row) else ''
                    
                    article_data = {}
                    
                    # Add the keyword and featured_img
                    if 'keyword' in row_dict:
                        article_data['keyword'] = row_dict['keyword']
                    if 'featured_img' in row_dict:
                        article_data['featured_img'] = row_dict['featured_img']
                    
                    # Process subtitle-image pairs
                    pairs = []
                    for i in range(1, 9):  # Support up to 8 subtitles
                        subtitle_key = f'Subtitle{i}'
                        img_key = f'img{i}'
                        
                        if subtitle_key in row_dict and row_dict[subtitle_key].strip():
                            subtitle = row_dict[subtitle_key].strip()
                            image = row_dict[img_key].strip() if img_key in row_dict else ''
                            
                            # Add the pair to both formats
                            pairs.append({'subtitle': subtitle, 'image': image})
                            
                            # Add to article_data in the traditional format
                            article_data[subtitle_key] = subtitle
                            article_data[img_key] = image
                    
                    # Store the structured pairs
                    article_data['subtitle_image_pairs'] = pairs
                    
                    # Only add articles that have at least a keyword
                    if article_data.get('keyword'):
                        self.data[index] = article_data

            return self.data

        except Exception as e:
            console.print(f"[red]Error processing structured file:[/] {str(e)}")
            return {}

    def validate_file(self) -> Tuple[bool, str]:
        """
        Validate the CSV file structure.

        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        console.rule("[bold blue]START CSV VALIDATION[/]", align="center")

        if not self.file_path.exists():
            console.print(f"[red]File not found:[/] {self.file_path}")
            return False, f"File not found: {self.file_path}"

        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)

                # Get headers
                try:
                    headers = next(reader)
                    console.print(f"[green]File found:[/] {self.file_path}")
                except StopIteration:
                    console.print("[red]CSV file is empty or has no headers[/]")
                    return False, "CSV file is empty or has no headers"

                # Create normalized headers mapping (lowercase for case-insensitive comparison)
                self.normalized_headers = {h.lower(): h for h in headers}

                # Display first line information
                if self.has_header:
                    console.print(f"[yellow]Header row:[/] {headers}")
                else:
                    console.print(f"[yellow]First line (not a header):[/] {headers}")

                console.print(f"[yellow]Number of columns:[/] {len(headers)}")

                # Check for required columns (case-insensitive)
                required_columns = [col.lower() for col in self.required_columns]
                missing_columns = [col for col in required_columns if col not in self.normalized_headers]

                if missing_columns:
                    # Instead of failing, try to detect if this is a simple keyword list
                    if len(headers) >= 1:
                        # If there's at least one column, assume it's the keyword column
                        self.simple_mode = True
                        console.print(f"[green]Extracting from column:[/] {headers[0]}")
                        return True, "Treating as simple keyword list"
                    else:
                        console.print("[red]Missing required columns[/]")
                        return False, self._format_missing_columns_error(missing_columns)

                # If we have the required columns, show which column we're extracting from
                for col in self.required_columns:
                    col_lower = col.lower()
                    if col_lower in self.normalized_headers:
                        console.print(f"[green]Extracting from column:[/] {self.normalized_headers[col_lower]}")

                # Count the number of data rows and validate required columns
                data_rows = []
                for row in reader:
                    # Create a dictionary mapping header names to values
                    row_dict = {}
                    for i, header in enumerate(headers):
                        row_dict[header] = row[i].strip() if i < len(row) else ''

                    # Verify required columns have values
                    valid_row = True
                    for col in self.required_columns:
                        col_lower = col.lower()
                        if col_lower in self.normalized_headers:
                            header = self.normalized_headers[col_lower]
                            if not row_dict[header].strip():
                                valid_row = False
                                break

                    if valid_row:
                        data_rows.append(row_dict)

                # Check if there's at least one valid row of data
                if not data_rows:
                    console.print("[red]No valid data rows found in CSV[/]")
                    return False, "No valid data rows found in CSV"

                console.print(f"\n[green]Total articles found:[/] {len(data_rows)}")

                # Create a table for the extracted articles
                table = Table(title="Extracted Articles", show_header=True)
                table.add_column("#", style="dim", width=4)

                if self.simple_mode:
                    table.add_column("Keyword", style="cyan")
                    table.add_column("Image Keyword", style="green")

                    # Display all articles (no limit)
                    for i, row_dict in enumerate(data_rows, 1):
                        if len(headers) >= 2:
                            table.add_row(str(i), row_dict[headers[0]], row_dict[headers[1]])
                        else:
                            table.add_row(str(i), row_dict[headers[0]], row_dict[headers[0]])
                else:
                    # For structured CSV, show keyword column and subtitle count
                    table.add_column("Keyword", style="cyan", width=30)
                    table.add_column("Subtitle Count", style="green", justify="center")

                    # Display all articles (no limit)
                    for i, row_dict in enumerate(data_rows, 1):
                        keyword_col = self.normalized_headers.get('keyword', headers[0])
                        keyword = row_dict[keyword_col]
                        
                        # Count non-empty subtitle columns
                        subtitle_count = sum(1 for key in row_dict if 'subtitle' in key.lower() and row_dict[key].strip())
                        
                        table.add_row(str(i), keyword, str(subtitle_count))

                console.print(table)
                console.rule("[bold blue]DONE CSV VALIDATION[/]", align="center")

                return True, "CSV file format is valid!"

        except Exception as e:
            console.print(f"[red]Error validating CSV:[/] {str(e)}")
            return False, f"Error validating CSV: {str(e)}"

    def _format_missing_columns_error(self, missing_columns: List[str]) -> str:
        """
        Format a helpful error message for missing columns.

        Args:
            missing_columns (List[str]): List of missing column names

        Returns:
            str: Formatted error message
        """
        message = f"""
ERROR: Your CSV file does not have the correct structure.

Missing required columns: {', '.join(missing_columns)}

The CSV file must have the following required columns: {', '.join(self.required_columns)}

For article generation with proper subheadings, your CSV should follow this format:
keyword,featured_img,subtitle1,img1,subtitle2,img2
how to grow tomatoes,tomato plant,Best Soil for Tomatoes,tomato soil,Watering Schedule,watering tomatoes

If you're using a simple keyword list without subheadings, it should look like:
keyword,image_keyword
how to grow tomatoes,tomato plant
how to make pasta,pasta cooking

Please correct your input file structure and try again.
"""
        return message

    def display_csv_template(self) -> None:
        """
        Display a template for the expected CSV format.
        """
        # Create a header
        console.rule("[bold blue]CSV FILE STRUCTURE GUIDE[/]", align="center")

        if self.simple_mode:
            # Create a simple template
            console.print("[yellow]SIMPLE MODE TEMPLATE[/]")
            console.print("\n[cyan]Expected simple CSV format:[/]")
            console.print("[bold]keyword,image_keyword[/]")

            console.print("\n[green]Example:[/]")
            console.print("how to grow tomatoes,tomato plant")
            console.print("how to make pasta,pasta cooking")

            console.print("\nThis format will generate articles without specific subheadings.")
        else:
            # Create a structured template
            console.print("[yellow]STRUCTURED MODE TEMPLATE[/]")
            console.print("\n[cyan]For full article generation with subheadings, use this format:[/]")
            console.print("[bold]keyword,featured_img,subtitle1,img1,subtitle2,img2,...[/]")

            console.print("\n[green]Example:[/]")
            console.print("how to grow tomatoes,tomato plant,Best Soil for Tomatoes,tomato soil,Watering Schedule,watering tomatoes")

            console.print("\nEach subtitle/image pair will create a section in your article.")
            console.print("You can add as many subtitle/image pairs as needed (subtitle3,img3, etc.)")

        # Create important notes
        console.print("\n[yellow]IMPORTANT NOTES[/]")
        console.print("1. The first row should contain the column headers exactly as shown above")
        console.print("2. Each row represents one article to be generated")
        console.print("3. The 'keyword' column is required and will be used as the main topic")
        console.print("4. For best results, ensure all subtitle/image pairs are properly matched")

        # Add a footer
        console.rule("[bold blue]END OF GUIDE[/]", align="center")

    def display_csv_structure(self) -> None:
        """
        Display the structure of the CSV file.
        """
        console.rule("[bold blue]CSV STRUCTURE ANALYSIS[/]", align="center")

        if self.simple_mode:
            console.print(f"[yellow]CSV STRUCTURE (SIMPLE MODE)[/]")
            console.print(f"[green]Total keywords:[/] {len(self.data)}")

            # Create a table for the keywords
            if self.data:
                table = Table(title="Keyword Preview", show_header=True)
                table.add_column("#", style="dim", width=4)
                table.add_column("Keyword", style="cyan")
                table.add_column("Image Keyword", style="green")

                # Show up to 5 keywords
                for i, (keyword, image_keyword) in enumerate(self.data[:5], 1):
                    table.add_row(str(i), keyword, image_keyword)

                console.print(table)
        else:
            console.print(f"[yellow]CSV STRUCTURE (STRUCTURED MODE)[/]")
            console.print(f"[green]Headers:[/] {list(self.normalized_headers.values())}")
            console.print(f"[green]Total articles:[/] {len(self.data)}")

            # Create a table for column mappings
            table = Table(title="Column Mappings", show_header=True)
            table.add_column("Config Column", style="cyan")
            table.add_column("Header Column", style="green")

            for config_col, header_col in self.normalized_headers.items():
                table.add_row(config_col, header_col)

            console.print(table)

            # If we have data, show a preview of the first article
            if self.data:
                first_article = next(iter(self.data.values()))

                preview_table = Table(title="First Article Preview", show_header=True)
                preview_table.add_column("Field", style="cyan")
                preview_table.add_column("Value", style="green")

                for key, value in first_article.items():
                    # Truncate long values
                    display_value = value
                    if len(str(value)) > 50:
                        display_value = str(value)[:47] + "..."
                    preview_table.add_row(key, display_value)

                console.print(preview_table)

        console.rule("[bold blue]END OF STRUCTURE ANALYSIS[/]", align="center")

    def get_article_data(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get article data for a specific index.

        Args:
            index (int): Article index (1-based)

        Returns:
            Optional[Dict[str, Any]]: Article data or None if not found
        """
        return self.data.get(index)

    def get_total_articles(self) -> int:
        """
        Get the total number of articles in the CSV.

        Returns:
            int: Total number of articles
        """
        return len(self.data)

    def get_all_keywords(self) -> List[str]:
        """
        Get all keywords from the CSV.

        Returns:
            List[str]: List of keywords
        """
        if self.simple_mode:
            return [item[0] for item in self.data]
        else:
            return [article_data.get('keyword', '') for article_data in self.data.values()]

    def validate_and_process(self) -> Tuple[bool, str]:
        """
        Validate and process the CSV file in one step.

        Returns:
            Tuple[bool, str]: (success, message)
        """
        # Detect file type
        if not self.detect_file_type():
            console.print("[red]Failed to detect file type[/]")
            return False, "Failed to detect file type"

        # Validate file
        is_valid, message = self.validate_file()
        if not is_valid:
            return False, message

        # Process file
        data = self.process_file()
        if not data:
            console.print("[red]Failed to process CSV file[/]")
            return False, "Failed to process CSV file"

        # Extract the keywords/articles for display
        if self.simple_mode:
            keywords = [item[0] for item in data] if isinstance(data, list) else []
            console.print(f"[green]Successfully processed {len(keywords)} keywords from CSV[/]")
            return True, f"Processed {len(keywords)} keywords from CSV"
        else:
            # Count the number of subtitle columns
            subtitle_count = 0
            for header in self.normalized_headers:
                if 'subtitle' in header or 'section' in header:
                    subtitle_count += 1
                    
            # Display the subtitle counts for each article if using flexible parsing
            if self.enable_flexible_parsing and isinstance(data, dict) and len(data) > 0:
                console.rule("[bold blue]ARTICLE SUBTITLE COUNT SUMMARY[/]", align="center")
                
                # Create a table for the subtitle counts
                table = Table(title="Article Subtitle Counts", show_header=True)
                table.add_column("#", style="dim", width=4)
                table.add_column("Article Keyword", style="cyan")
                table.add_column("Number of Subtitles", style="green")
                
                # Add a row for each article
                for index, article_data in data.items():
                    # Count subtitles in this article
                    article_subtitle_count = sum(1 for key in article_data if key.startswith("subtitle") and article_data[key])
                    keyword = article_data.get('keyword', 'Unknown')
                    table.add_row(str(index), keyword, str(article_subtitle_count))
                
                console.print(table)
                console.rule("[bold blue]END OF SUBTITLE COUNT SUMMARY[/]", align="center")

            console.print(f"[green]Successfully processed {len(data)} articles with {subtitle_count} subheadings per article[/]")
            return True, f"Processed {len(data)} articles from CSV with {subtitle_count} subheadings per article"

    def _detect_dynamic_subtitle_columns(self, headers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Dynamically detect subtitle and image columns from CSV headers.
        
        This method scans headers for patterns like subtitle1, subtitle2, sub1, sub2, etc.
        and their corresponding image columns like img1, image1, pic1, etc.
        
        Args:
            headers (List[str]): List of column headers from the CSV file
            
        Returns:
            Tuple[List[str], List[str]]: Tuple of (subtitle_columns, image_columns)
        """
        subtitle_columns = []
        image_columns = []
        
        # Create mappings of header indices for easier matching
        subtitle_indices = {}  # {index_number: header_name}
        image_indices = {}  # {index_number: header_name}
        
        # Process headers to find potential subtitle and image columns
        for header in headers:
            header_lower = header.lower().strip()
            
            # Check for subtitle patterns
            for pattern in self.subtitle_patterns:
                if pattern in header_lower:
                    # Try to extract the index number
                    index_str = ''.join(filter(str.isdigit, header_lower))
                    if index_str:
                        index = int(index_str)
                        subtitle_indices[index] = header
                        break
            
            # Check for image patterns
            for pattern in self.image_patterns:
                if pattern in header_lower:
                    # Try to extract the index number
                    index_str = ''.join(filter(str.isdigit, header_lower))
                    if index_str:
                        index = int(index_str)
                        image_indices[index] = header
                        break
        
        # Sort columns by index number to maintain order
        subtitle_columns = [subtitle_indices[i] for i in sorted(subtitle_indices.keys())]
        image_columns = [image_indices[i] for i in sorted(image_indices.keys())]
        
        console.print(f"[green]Dynamically detected subtitle columns:[/] {subtitle_columns}")
        console.print(f"[green]Dynamically detected image columns:[/] {image_columns}")
        
        return subtitle_columns, image_columns

    def _get_flexible_column_mapping(self, headers: List[str]) -> Dict[str, Any]:
        """
        Create flexible column mapping that adapts to different CSV structures.
        
        This method dynamically detects required columns, optional columns, and
        subtitle/image column pairs from the headers.
        
        Args:
            headers (List[str]): List of column headers from the CSV file
            
        Returns:
            Dict[str, Any]: Dictionary with column mappings containing:
                - required_cols_map: Dictionary mapping required column names to actual header names
                - optional_cols_map: Dictionary mapping optional column names to actual header names
                - subtitle_cols_map: Dictionary mapping subtitle column names to actual header names
                - image_cols_map: Dictionary mapping image column names to actual header names
                - subtitle_image_pairs: List of (subtitle_header, image_header) pairs
        """
        # Create case-insensitive mappings for required columns
        required_cols_map = {}
        for col in self.required_columns:
            for header_key, header_val in self.normalized_headers.items():
                if header_key == col.lower():
                    required_cols_map[col] = header_val
        
        # Create case-insensitive mappings for optional columns
        optional_cols_map = {}
        for col in self.optional_columns:
            for header_key, header_val in self.normalized_headers.items():
                if header_key == col.lower():
                    optional_cols_map[col] = header_val
        
        # Determine if we should use dynamic detection or fixed columns
        if self.enable_flexible_parsing:
            # Dynamically detect subtitle and image columns
            subtitle_cols, image_cols = self._detect_dynamic_subtitle_columns(headers)
            
            # If dynamic detection found columns, use those
            if subtitle_cols:
                self.dynamic_subtitle_columns = subtitle_cols
                self.dynamic_image_columns = image_cols
            else:
                # Fall back to config-defined columns
                subtitle_cols = self.subtitle_columns
                image_cols = self.image_columns
        else:
            # Use config-defined columns
            subtitle_cols = self.subtitle_columns
            image_cols = self.image_columns
        
        # Create mappings for subtitle columns
        subtitle_cols_map = {}
        for col in subtitle_cols:
            col_lower = col.lower()
            for header_key, header_val in self.normalized_headers.items():
                if header_key == col_lower or header_val == col:
                    subtitle_cols_map[col] = header_val
                    break
        
        # Create mappings for image columns
        image_cols_map = {}
        for col in image_cols:
            col_lower = col.lower()
            for header_key, header_val in self.normalized_headers.items():
                if header_key == col_lower or header_val == col:
                    image_cols_map[col] = header_val
                    break
        
        # Create subtitle-image pairs for easier processing
        subtitle_image_pairs = []
        
        # First try to match by index number
        subtitle_indices = {}
        image_indices = {}
        
        # Extract indices from subtitle columns
        for subtitle_col in subtitle_cols:
            subtitle_col_lower = subtitle_col.lower()
            index_match = re.search(r'\d+', subtitle_col_lower)
            if index_match:
                index_num = int(index_match.group())
                subtitle_indices[index_num] = subtitle_col
        
        # Extract indices from image columns
        for image_col in image_cols:
            image_col_lower = image_col.lower()
            index_match = re.search(r'\d+', image_col_lower)
            if index_match:
                index_num = int(index_match.group())
                image_indices[index_num] = image_col
        
        # Match subtitle and image columns by index
        for index_num in sorted(subtitle_indices.keys()):
            subtitle_col = subtitle_indices[index_num]
            image_col = image_indices.get(index_num)  # May be None if no matching image column
            
            if subtitle_col in subtitle_cols_map:
                subtitle_header = subtitle_cols_map[subtitle_col]
                image_header = image_cols_map.get(image_col) if image_col in image_cols_map else None
                subtitle_image_pairs.append((subtitle_header, image_header))
        
        return {
            "required_cols_map": required_cols_map,
            "optional_cols_map": optional_cols_map,
            "subtitle_cols_map": subtitle_cols_map,
            "image_cols_map": image_cols_map,
            "subtitle_image_pairs": subtitle_image_pairs
        }

    def _extract_subtitle_image_pairs(self, row: Dict[str, str], subtitle_image_pairs: List[Tuple[str, Optional[str]]]) -> List[Dict[str, str]]:
        """
        Extract subtitle/image pairs from a single row, handling variable numbers.
        
        This method processes a row of data and extracts all non-empty subtitle/image pairs,
        allowing for varying numbers of subtitles per article.
        
        Args:
            row (Dict[str, str]): Dictionary mapping column names to values for a single row
            subtitle_image_pairs (List[Tuple[str, Optional[str]]]): List of (subtitle_header, image_header) pairs
            
        Returns:
            List[Dict[str, str]]: List of subtitle/image pair dictionaries, each containing:
                - subtitle: The subtitle text
                - image: The image URL (or None if no image)
        """
        result = []
        
        # Process each subtitle (Subtitle1, Subtitle2, etc.)
        for i in range(1, 9):  # Support up to 8 subtitles
            subtitle_key = f'Subtitle{i}'
            img_key = f'img{i}'
            
            # Check if this subtitle exists and has content
            if subtitle_key in row and row[subtitle_key].strip():
                subtitle = row[subtitle_key].strip()
                image = row[img_key].strip() if img_key in row else ''
                
                result.append({
                    'subtitle': subtitle,
                    'image': image
                })
            else:
                # Stop when we find the first empty subtitle
                # This handles rows with varying numbers of subtitles
                break
        
        return result
