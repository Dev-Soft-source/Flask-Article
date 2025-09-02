# Test script for external_links_handler.py

import os
import sys
from article_generator.external_links_handler import generate_external_links_section
from config import Config

# Create a minimal config for testing
class TestConfig(Config):
    def __init__(self):
        # Set required properties for the external links handler
        self.add_external_links_into_article = True
        self.serp_api_key = "bc8dbc71b17e92497cc341bf1e0a476c4c9d0dc7683cceac0779744709287cf7"  # Use your actual key

def main():
    print("Testing external links handler")
    
    # Create test config
    config = TestConfig()
    
    # Test keywords
    test_keywords = [
        "python programming",
        "artificial intelligence",
        "machine learning tutorials"
    ]
    
    for keyword in test_keywords:
        print(f"\nTesting keyword: {keyword}")
        
        # Test WordPress format
        print("Testing WordPress format...")
        result = generate_external_links_section(
            keyword=keyword,
            config=config,
            output_format='wordpress'
        )
        
        if result:
            print(f"Success! Generated WordPress content with length: {len(result)}")
            print(f"First 100 chars: {result[:100]}...")
        else:
            print("Failed to generate WordPress content")
        
        # Test Markdown format
        print("\nTesting Markdown format...")
        result = generate_external_links_section(
            keyword=keyword,
            config=config,
            output_format='markdown'
        )
        
        if result:
            print(f"Success! Generated Markdown content with length: {len(result)}")
            print(f"First 100 chars: {result[:100]}...")
        else:
            print("Failed to generate Markdown content")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main() 