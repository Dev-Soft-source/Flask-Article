#!/usr/bin/env python3
"""
Test script to verify hierarchical structure support in article generation.

This script tests both 2-level and 3-level hierarchy support.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'script1'))

from script1.article_generator.content_generator import generate_section
from script1.article_generator.text_processor import convert_wp_to_markdown
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockArticleContext:
    """Mock context for testing."""
    
    def __init__(self):
        self.config = ArticleConfig()
        self.section_token_limit = 1000
        self.paragraphs_per_section = 3
        self.min_paragraph_tokens = 100
        self.max_paragraph_tokens = 300
        self.voicetone = "professional"
        self.articletype = "informative"
        self.articlelanguage = "English"
        self.articleaudience = "general"
        self.pointofview = "third person"
        self.article_parts = {"sections": []}

def test_2_level_hierarchy():
    """Test 2-level hierarchy (sections -> paragraphs)."""
    print("Testing 2-level hierarchy...")
    
    context = MockArticleContext()
    parsed_sections = [
        {
            "title": "Introduction to AI",
            "subsections": ["What is AI", "History of AI", "Current applications"]
        },
        {
            "title": "Machine Learning Basics",
            "subsections": ["Supervised learning", "Unsupervised learning", "Reinforcement learning"]
        }
    ]
    
    # Test section generation
    result = generate_section(
        context=context,
        heading="Introduction to AI",
        keyword="artificial intelligence",
        section_number=1,
        total_sections=2,
        paragraph_prompt="Write about {keyword} focusing on {heading}",
        parsed_sections=parsed_sections
    )
    
    print("2-level hierarchy result:")
    print(result)
    print("-" * 50)
    
    # Verify structure
    assert "## Introduction to AI" in result
    assert "<p>" in result
    return True

def test_3_level_hierarchy():
    """Test 3-level hierarchy (sections -> subsections -> paragraphs)."""
    print("Testing 3-level hierarchy...")
    
    context = MockArticleContext()
    parsed_sections = [
        {
            "title": "Machine Learning",
            "subsections": [
                "Supervised Learning",
                "Unsupervised Learning", 
                "Deep Learning"
            ]
        }
    ]
    
    # Test section generation with subsections
    result = generate_section(
        context=context,
        heading="Machine Learning",
        keyword="machine learning",
        section_number=1,
        total_sections=1,
        paragraph_prompt="Write about {keyword} focusing on {heading}",
        parsed_sections=parsed_sections
    )
    
    print("3-level hierarchy result:")
    print(result)
    print("-" * 50)
    
    # Verify structure
    assert "## Machine Learning" in result
    return True

def test_wordpress_formatting():
    """Test WordPress formatting with hierarchical structure."""
    print("Testing WordPress formatting...")
    
    from script1.article_generator.text_processor import format_article_for_wordpress
    
    article_dict = {
        'title': 'Test Article',
        'introduction': '<p>This is an introduction</p>',
        'sections': [
            '## Main Section\n\n<h3>Subsection 1</h3>\n<p>Content for subsection 1</p>\n\n<h3>Subsection 2</h3>\n<p>Content for subsection 2</p>',
            '## Another Section\n\n<h4>Deep Subsection</h4>\n<p>Content for deep subsection</p>'
        ],
        'conclusion': '<p>This is a conclusion</p>',
        'summary': 'This is a summary',
        'block_notes': 'These are key takeaways'
    }
    
    config = ArticleConfig()
    config.add_summary_into_article = True
    config.add_blocknotes = True
    
    result = format_article_for_wordpress(article_dict, config)
    
    print("WordPress formatting result:")
    print(result[:500] + "..." if len(result) > 500 else result)
    print("-" * 50)
    
    # Verify WordPress structure
    assert '<!-- wp:heading -->' in result
    assert '<h2>' in result
    assert '<h3>' in result or '<h4>' in result
    return True

def test_markdown_formatting():
    """Test Markdown formatting with hierarchical structure."""
    print("Testing Markdown formatting...")
    
    wp_content = '''<!-- wp:heading -->
<h2>Main Section</h2>
<!-- /wp:heading -->
<!-- wp:heading {"level":3} -->
<h3>Subsection</h3>
<!-- /wp:heading -->
<!-- wp:heading {"level":4} -->
<h4>Deep Subsection</h4>
<!-- /wp:heading -->'''
    
    result = convert_wp_to_markdown(wp_content)
    
    print("Markdown formatting result:")
    print(result)
    print("-" * 50)
    
    # Verify Markdown structure
    assert '## Main Section' in result
    assert '### Subsection' in result
    assert '#### Deep Subsection' in result
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("HIERARCHICAL STRUCTURE TEST SUITE")
    print("=" * 60)
    
    try:
        test_2_level_hierarchy()
        test_3_level_hierarchy()
        test_wordpress_formatting()
        test_markdown_formatting()
        
        print("✅ All hierarchical structure tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)