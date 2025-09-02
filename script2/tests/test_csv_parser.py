# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import os
import sys
from pathlib import Path
from utils.csv_utils import CSVProcessor
from config import Config
from utils.rich_provider import provider

def test_csv_parser(csv_file_path: str) -> None:
    """
    Test the CSV parser with the given file.

    Args:
        csv_file_path: Path to the CSV file to test
    """
    provider.info(f"Testing CSV parser with file: {csv_file_path}")

    # Initialize configuration with test settings
    config = Config(
        # Disable features that require API keys
        add_paa_paragraphs_into_article=False,
        enable_rag=False,
        add_youtube_video=False,
        enable_image_generation=False,
        enable_wordpress_upload=False
    )

    # Initialize CSV processor
    csv_processor = CSVProcessor(csv_file_path, config)

    # Validate file
    is_valid, message = csv_processor.validate_file()

    if is_valid:
        provider.success(f"CSV validation successful: {message}")

        # Process file
        data = csv_processor.process_file()

        if data:
            provider.success(f"CSV processing successful. Found {len(data)} articles.")

            # Display CSV structure
            csv_processor.display_csv_structure()

            # Display first article data
            first_article = csv_processor.get_article_data(1)
            if first_article:
                provider.info("First article data:")
                for key, value in first_article.items():
                    provider.info(f"  {key}: {value[:50]}..." if len(str(value)) > 50 else f"  {key}: {value}")
        else:
            provider.error("CSV processing failed.")
    else:
        provider.error(f"CSV validation failed: {message}")
        csv_processor.display_csv_template()

def create_test_csv(output_path: str, valid: bool = True) -> None:
    """
    Create a test CSV file with various formats to test the parser.

    Args:
        output_path: Path to save the test CSV file
        valid: Whether to create a valid or invalid CSV file
    """
    provider.info(f"Creating {'valid' if valid else 'invalid'} test CSV file at: {output_path}")

    if valid:
        # Create valid test CSV content
        csv_content = """keyword,featured_img,Subtitle1,img1,Subtitle2,img2,Subtitle3,img3
How to grow tomatoes,tomato plant,Best Soil for Tomatoes,tomato soil,Watering Schedule,watering tomatoes,Common Pests,tomato pests
KEYWORD with UPPERCASE,test image,SUBTITLE with UPPERCASE,test image 1,Another Subtitle,test image 2,,
Keyword with missing image,,Subtitle without image,,,,,,
"""
    else:
        # Create invalid test CSV content with missing required columns and mismatched subtitle/image pairs
        csv_content = """title,description,section1,image1,section2
How to grow tomatoes,A guide to growing tomatoes,Soil preparation,tomato soil,Watering tips
Growing herbs,Herb gardening basics,Types of herbs,herb garden,Harvesting herbs
"""

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(csv_content)

    provider.success(f"Test CSV file created at: {output_path}")

def main() -> int:
    """Main entry point for the CSV parser test."""
    try:
        provider.info("CSV Parser Test Utility")

        # Check if a CSV file path was provided
        if len(sys.argv) > 1:
            csv_file_path = sys.argv[1]

            # Check if the file exists
            if not os.path.exists(csv_file_path):
                provider.error(f"File not found: {csv_file_path}")
                return 1

            # Test the CSV parser
            test_csv_parser(csv_file_path)
        else:
            # Create a test CSV file
            test_csv_path = "test_input.csv"
            create_test_csv(test_csv_path)

            # Test the CSV parser with the test file
            test_csv_parser(test_csv_path)

        return 0

    except Exception as e:
        provider.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
