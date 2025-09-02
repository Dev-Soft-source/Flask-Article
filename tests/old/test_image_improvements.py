#!/usr/bin/env python3
# Test script for image handling improvements in CopyscriptAI
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import os
import sys
import time
import argparse
from PIL import Image
import random

# Add the parent directory to the Python path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import logger
from script1.article_generator.logger import logger

def test_image_compression():
    """Test image compression functionality."""
    logger.info("Testing image compression...")
    
    # Create test directory
    test_dir = os.path.join(os.path.dirname(__file__), "test_images")
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a test image
    test_image_path = os.path.join(test_dir, "test_image.jpg")
    
    # Check if test image exists, if not create one
    if not os.path.exists(test_image_path):
        # Create a simple test image
        img = Image.new('RGB', (1000, 1000), color='blue')
        img.save(test_image_path, quality=100)
        logger.info(f"Created test image: {test_image_path}")
    
    # Get original size
    original_size = os.path.getsize(test_image_path)
    logger.info(f"Original image size: {original_size} bytes")
    
    # Test compression with different quality settings
    quality_levels = [90, 70, 50, 30, 10]
    
    for quality in quality_levels:
        # Create copy of the original image
        compressed_path = os.path.join(test_dir, f"test_image_q{quality}.jpg")
        
        # Open and compress
        with Image.open(test_image_path) as img:
            img.save(compressed_path, format='JPEG', optimize=True, quality=quality)
        
        # Get compressed size
        compressed_size = os.path.getsize(compressed_path)
        reduction = (1 - (compressed_size / original_size)) * 100
        
        logger.info(f"Quality {quality}: {compressed_size} bytes ({reduction:.1f}% reduction)")
    
    logger.success("Image compression test completed")

def test_image_alignment():
    """Test image alignment parameters."""
    logger.info("Testing image alignment...")
    
    # Import image handler from script1
    from script1.article_generator.image_handler import ImageConfig, process_body_image
    
    # Test alignment parameters
    alignment_options = ["aligncenter", "alignleft", "alignright"]
    
    for alignment in alignment_options:
        # Create test image data
        image_data = {
            "url": "https://example.com/test.jpg",
            "alt": "Test image",
            "photographer": "Test Photographer",
            "photographer_url": "https://example.com"
        }
        
        # Process with specific alignment
        processed = process_body_image(
            image_data=image_data,
            keyword="test",
            index=1,
            alignment=alignment
        )
        
        # Verify alignment was set correctly
        if processed and processed.get("alignment") == alignment:
            logger.success(f"Alignment {alignment} correctly set in processed image")
        else:
            logger.error(f"Alignment {alignment} not correctly set in processed image")
    
    logger.success("Image alignment test completed")

def test_duplicate_prevention():
    """Test duplicate image prevention."""
    logger.info("Testing duplicate image prevention...")
    
    # Import from script1
    from script1.article_generator.image_handler import ImageConfig, get_article_images
    
    # Create a test config with duplicate prevention enabled
    config = ImageConfig(
        enable_image_generation=True,
        randomize_images=False,
        max_number_of_images=5,
        prevent_duplicate_images=True,
        image_api_key=os.environ.get("IM_API_KEY", "")
    )
    
    # Skip test if no API key
    if not config.image_api_key:
        logger.warning("Skipping duplicate prevention test - No IMAGE API key found")
        return
    
    # Test with a keyword that should return multiple images
    keyword = "nature"
    
    # Get images with duplicate prevention enabled
    feature_image, body_images = get_article_images(keyword, config)
    
    # Check if we got any images
    if not feature_image or not body_images:
        logger.warning("No images found - can't test duplicate prevention")
        return
    
    # Get unique image IDs
    image_ids = set()
    
    # Add feature image ID
    feature_url = feature_image.get("url")
    
    # Add body image IDs
    for img in body_images:
        image_ids.add(img.get("url"))
    
    # Add feature image ID
    if feature_url:
        image_ids.add(feature_url)
    
    # Check if all images are unique
    logger.info(f"Total images: {len(body_images) + 1}, Unique images: {len(image_ids)}")
    
    if len(image_ids) == len(body_images) + 1:
        logger.success("All images are unique - duplicate prevention working correctly")
    else:
        logger.error("Duplicate images found - prevention not working correctly")
    
    logger.success("Duplicate prevention test completed")

def main():
    """Main function to run all tests."""
    parser = argparse.ArgumentParser(description="Test image handling improvements")
    parser.add_argument("--compression", action="store_true", help="Test image compression")
    parser.add_argument("--alignment", action="store_true", help="Test image alignment")
    parser.add_argument("--duplication", action="store_true", help="Test duplicate prevention")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    # Run all tests if --all is specified or no specific test is specified
    run_all = args.all or not (args.compression or args.alignment or args.duplication)
    
    if args.compression or run_all:
        test_image_compression()
    
    if args.alignment or run_all:
        test_image_alignment()
    
    if args.duplication or run_all:
        test_duplicate_prevention()
    
    logger.success("All tests completed!")

if __name__ == "__main__":
    main()
