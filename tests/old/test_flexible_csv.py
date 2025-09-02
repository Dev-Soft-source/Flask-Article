# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import os
import sys
from pathlib import Path
from config import Config
from utils.rich_provider import provider

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # project root directory
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import UnifiedCSVProcessor directly
from utils.unified_csv_processor import UnifiedCSVProcessor
from utils.csv_utils import CSVProcessor

def test_flexible_csv_parsing():
    """
    Test the flexible CSV parsing functionality with a sample CSV file.
    """
    provider.info("Testing Flexible CSV Parsing")
    
    # Load configuration
    config = Config()
    
    # Ensure flexible parsing is enabled
    config.CSV_FLEXIBLE_PARSING = True
    
    # Path to test CSV file
    test_csv_path = os.path.join(script_dir, "test_flexible_input.csv")
    
    # Create UnifiedCSVProcessor instance
    provider.info("Testing UnifiedCSVProcessor directly...")
    unified_processor = UnifiedCSVProcessor(test_csv_path, config)
    
    # Validate and process the file
    is_valid, message = unified_processor.validate_file()
    if is_valid:
        provider.success(f"CSV validation successful: {message}")
        
        # Process the file
        result = unified_processor.process_file()
        
        # Print the results
        provider.info(f"Processed {len(result)} articles with flexible structure")
        
        # Display article details
        for index, article_data in result.items():
            provider.info(f"Article {index}: {article_data.get('keyword')}")
            
            # Count subtitle/image pairs
            subtitle_count = 0
            for key in article_data:
                if key.startswith("subtitle") and article_data[key]:
                    subtitle_count += 1
            
            provider.info(f"  - Has {subtitle_count} subtitle/image pairs")
            
            # Print subtitle/image pairs if available
            if "subtitle_image_pairs" in article_data:
                provider.info(f"  - Flexible structure detected with {len(article_data['subtitle_image_pairs'])} pairs")
                for i, pair in enumerate(article_data["subtitle_image_pairs"], 1):
                    provider.info(f"    {i}. {pair['subtitle']} -> {pair['image'] or 'No image'}")
    else:
        provider.error(f"CSV validation failed: {message}")
    
    # Test with CSVProcessor
    provider.info("\nTesting CSVProcessor wrapper...")
    csv_processor = CSVProcessor(test_csv_path, config)
    
    # Validate and process the file
    is_valid, message = csv_processor.validate_file()
    if is_valid:
        provider.success(f"CSV validation successful: {message}")
        
        # Process the file
        result = csv_processor.process_file()
        
        # Print the results
        provider.info(f"Processed {len(result)} articles with flexible structure")
        
        # Display article details
        for index, article_data in result.items():
            provider.info(f"Article {index}: {article_data.get('keyword')}")
            
            # Count subtitle/image pairs
            subtitle_count = 0
            for key in article_data:
                if key.startswith("subtitle") and article_data[key]:
                    subtitle_count += 1
            
            provider.info(f"  - Has {subtitle_count} subtitle/image pairs")
    else:
        provider.error(f"CSV validation failed: {message}")

if __name__ == "__main__":
    test_flexible_csv_parsing()
