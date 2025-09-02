# CSV Parsing Improvements

## Overview

This document describes the improvements made to the CSV parsing functionality in the CopyscriptAI article generation system. The improvements address the issues identified in the CSV parsing analysis, providing a more consistent and user-friendly experience while maintaining compatibility with existing files.

## Key Improvements

### 1. Unified CSV Processor

A new `UnifiedCSVProcessor` class has been created that can handle both simple keyword lists (script1 style) and structured CSV files (script2 style). This provides a consistent interface for CSV parsing across the codebase.

Key features:
- Automatic detection of file type (simple or structured)
- Improved header detection
- Case-insensitive matching
- Flexible column mapping
- Better error handling and feedback

### 2. Improved Header Detection

The CSV parser now automatically detects if the first row is a header based on content analysis. This allows it to correctly handle both files with and without headers.

```python
def _detect_header(self, first_line: str) -> bool:
    """
    Detect if the first line is a header.

    Args:
        first_line (str): The first line of the file

    Returns:
        bool: True if the first line appears to be a header, False otherwise
    """
    first_line_lower = first_line.lower()

    # Check for common header keywords
    header_indicators = ['keyword', 'title', 'featured', 'image', 'subtitle', 'img']
    for indicator in header_indicators:
        if indicator in first_line_lower:
            return True

    # Additional heuristics for header detection
    # ...
```

### 3. Case-Insensitive Matching

The CSV parser now uses case-insensitive matching for both headers and data, ensuring that uppercase and lowercase variations are handled consistently.

```python
# Create normalized headers mapping (lowercase for case-insensitive comparison)
self.normalized_headers = {h.lower(): h for h in headers}

# Check for required columns (case-insensitive)
required_columns = [col.lower() for col in self.required_columns]
```

### 4. Flexible Column Mapping

The CSV parser now supports more flexible column mapping, allowing it to match columns by similarity rather than exact match. This is particularly useful for structured CSV files with different column names.

```python
# Match by number (e.g., subtitle1 matches Subtitle1, SUBTITLE1, etc.)
col_num = re.search(r'\d+', col)
header_num = re.search(r'\d+', header_key)

if col_num and header_num and col_num.group() == header_num.group() and 'subtitle' in header_key:
    subtitle_cols_map[col] = header_val
```

### 5. Better Error Handling

The CSV parser now provides more detailed error messages and suggestions for fixing issues, making it easier for users to understand and resolve problems.

```python
def _format_missing_columns_error(self, missing_columns: List[str]) -> str:
    """
    Format a helpful error message for missing columns.

    Args:
        missing_columns (List[str]): List of missing column names

    Returns:
        str: Formatted error message
    """
    message = f"""
CSV must have the following required columns: {', '.join(self.required_columns)}

Example of valid CSV format:
keyword,featured_img,subtitle1,img1,subtitle2,img2
how to grow tomatoes,tomato plant,Best Soil for Tomatoes,tomato soil,Watering Schedule,watering tomatoes

If you're using a simple keyword list, it should look like:
keyword,image_keyword
how to grow tomatoes,tomato plant
how to make pasta,pasta cooking
"""
    return message
```

## Integration with Existing Code

### Script1 Integration

The `read_keywords_file` function in script1 has been updated to use the `UnifiedCSVProcessor`, providing improved CSV parsing capabilities while maintaining backward compatibility.

```python
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
    # Initialize the unified CSV processor
    processor = UnifiedCSVProcessor(file_path)

    # Detect file type and process
    processor.detect_file_type()

    # Process the file
    result = processor.process_file()

    # Handle both simple and structured CSV formats
    # ...
```

### Script2 Integration

The `CSVProcessor` class in script2 has been updated to use the `UnifiedCSVProcessor` internally, providing improved CSV parsing capabilities while maintaining the existing interface.

```python
class CSVProcessor:
    """
    Handles CSV file processing for article generation with improved validation and error handling.

    This class now uses the UnifiedCSVProcessor internally to provide better compatibility
    with different CSV formats, improved header detection, and case-insensitive matching.
    """

    def __init__(self, file_path: str, config: Config):
        # ...
        # Initialize the unified CSV processor
        self.unified_processor = UnifiedCSVProcessor(file_path, config)

    def validate_file(self) -> Tuple[bool, str]:
        # Use the unified processor for validation
        is_valid, message = self.unified_processor.validate_file()
        # ...

    def process_file(self) -> Dict[int, Dict[str, Any]]:
        # Use the unified processor to process the file
        result = self.unified_processor.process_file()
        # ...
```

## Testing

A comprehensive test suite has been created to verify that the CSV parsing improvements work correctly with various file formats:

- Simple CSV files without headers
- Simple CSV files with headers
- Structured CSV files with standard column names
- Structured CSV files with different column names

The tests verify that both script1 and script2 can correctly parse all these file formats, ensuring backward compatibility and improved functionality.

### Test Suite

The test suite uses Python's `unittest` framework to provide proper assertions and validation:

```python
class TestCSVParsing(unittest.TestCase):
    """Test case for CSV parsing functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test files once before all tests."""
        cls.create_test_files()

    @classmethod
    def tearDownClass(cls):
        """Clean up test files after all tests."""
        cls.cleanup_test_files()

    def test_script1_simple_csv(self):
        """Test script1's CSV parsing with a simple CSV file."""
        keywords = read_keywords_file('test_simple.csv')

        # Assertions
        self.assertEqual(len(keywords), 2, "Should parse 2 keywords from simple CSV")
        self.assertEqual(keywords[0][0], "how to make pasta", "First keyword should match")
        self.assertEqual(keywords[0][1], "pasta cooking", "First image keyword should match")
        self.assertEqual(keywords[1][0], "KEYWORD with UPPERCASE", "Case should be preserved")

    # Additional test methods...
```

The test suite includes tests for:

1. **Script1 CSV Parsing**:
   - Simple CSV files
   - Simple CSV files with headers
   - Structured CSV files
   - Structured CSV files with different column names

2. **Script2 CSV Parsing**:
   - Simple CSV files
   - Simple CSV files with headers
   - Structured CSV files
   - Structured CSV files with different column names

Each test includes proper assertions to verify that the CSV parsing works correctly, including checking the number of keywords/articles parsed and verifying the content of the parsed data.

## Conclusion

The CSV parsing improvements provide a more consistent and user-friendly experience while maintaining compatibility with existing files. The unified approach ensures that both simple keyword lists and structured CSV files are handled correctly, with improved header detection, case-insensitive matching, and flexible column mapping.

These improvements address the issues identified in the CSV parsing analysis and provide a solid foundation for future enhancements to the CopyscriptAI article generation system.
