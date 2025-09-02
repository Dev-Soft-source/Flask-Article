import pytest
import re
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from Script1
from script1.article_generator.content_generator import generate_paragraph, ArticleContext
from script1.article_generator.text_processor import parse_paragraph_with_heading, convert_to_markdown

# Test data
MOCK_GOOD_RESPONSE = """<paragraph>Machine learning is a subset of artificial intelligence that involves the development of algorithms that allow computers to learn from and make decisions based on data.</paragraph>
<heading>Understanding Machine Learning</heading>"""

MOCK_LEGACY_RESPONSE = """[CONTENT] Machine learning is a subset of artificial intelligence that involves the development of algorithms that allow computers to learn from and make decisions based on data.
[HEADING] Understanding Machine Learning"""

MOCK_BAD_RESPONSE = """Here's a paragraph about machine learning:
Machine learning is a subset of artificial intelligence that involves the development of algorithms that allow computers to learn from and make decisions based on data."""

MOCK_MISSING_CONTENT = """<heading>Understanding Machine Learning</heading>"""

MOCK_MISSING_HEADING = """<paragraph>Machine learning is a subset of artificial intelligence.</paragraph>"""

MOCK_LEGACY_MISSING_CONTENT = """[HEADING] Understanding Machine Learning"""

MOCK_LEGACY_MISSING_HEADING = """[CONTENT] Machine learning is a subset of artificial intelligence."""

class TestParagraphHeadings:
    """Test cases for paragraph headings functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        # Mock ArticleContext
        self.mock_context = MagicMock(spec=ArticleContext)
        self.mock_context.config.content_generation_temperature = 0.7
        self.mock_context.config.token_limits = {'paragraph': 500}
        self.mock_context.voicetone = "professional"
        self.mock_context.articletype = "informational"
        self.mock_context.articlelanguage = "english"
        self.mock_context.articleaudience = "general"
        self.mock_context.pointofview = "third person"
        self.mock_context.paragraphs_per_section = 3
    
    @patch('script1.article_generator.content_generator.gpt_completion')
    def test_generate_paragraph_with_heading_success(self, mock_gpt):
        """Test successful paragraph with heading generation."""
        mock_gpt.return_value = MOCK_GOOD_RESPONSE
        
        result = generate_paragraph(
            context=self.mock_context,
            heading="Introduction to Machine Learning",
            keyword="machine learning",
            current_paragraph=1,
            paragraphs_per_section=3,
            section_number=1,
            total_sections=5,
            section_points=["History of machine learning", "Applications of ML"]
        )
        
        # Check if result contains both heading and content in HTML format
        assert "<h4>Understanding Machine Learning</h4>" in result
        assert "<p>Machine learning is a subset of artificial intelligence" in result
    
    @patch('script1.article_generator.content_generator.gpt_completion')
    def test_generate_paragraph_with_heading_malformed_response(self, mock_gpt):
        """Test handling of malformed response."""
        mock_gpt.return_value = MOCK_BAD_RESPONSE
        
        result = generate_paragraph(
            context=self.mock_context,
            heading="Introduction to Machine Learning",
            keyword="machine learning",
            current_paragraph=1,
            paragraphs_per_section=3
        )
        
        # Check if result contains a generic heading with the paragraph content
        assert "<h4>About machine learning</h4>" in result
        assert "<p>Here's a paragraph about machine learning" in result
    
    @patch('script1.article_generator.content_generator.gpt_completion')
    def test_generate_paragraph_with_heading_missing_content(self, mock_gpt):
        """Test handling of response with missing content."""
        mock_gpt.return_value = MOCK_MISSING_CONTENT
        
        result = generate_paragraph(
            context=self.mock_context,
            heading="Introduction to Machine Learning",
            keyword="machine learning"
        )
        
        # Check if result contains a generic heading with the paragraph content
        assert "<h4>About machine learning</h4>" in result
    
    @patch('script1.article_generator.content_generator.gpt_completion')
    def test_generate_paragraph_with_heading_missing_heading(self, mock_gpt):
        """Test handling of response with missing heading."""
        mock_gpt.return_value = MOCK_MISSING_HEADING
        
        result = generate_paragraph(
            context=self.mock_context,
            heading="Introduction to Machine Learning",
            keyword="machine learning"
        )
        
        # Check if result contains a generic heading with the paragraph content
        assert "<h4>About machine learning</h4>" in result
        assert "<p>Machine learning is a subset of artificial intelligence.</p>" in result
    
    @patch('script1.article_generator.content_generator.gpt_completion')
    def test_generate_paragraph_with_heading_legacy_format(self, mock_gpt):
        """Test successful paragraph with heading generation using legacy format."""
        mock_gpt.return_value = MOCK_LEGACY_RESPONSE
        
        result = generate_paragraph(
            context=self.mock_context,
            heading="Introduction to Machine Learning",
            keyword="machine learning",
            current_paragraph=1,
            paragraphs_per_section=3,
            section_number=1,
            total_sections=5,
            section_points=["History of machine learning", "Applications of ML"]
        )
        
        # Check if result contains both heading and content in HTML format
        assert "<h4>Understanding Machine Learning</h4>" in result
        assert "<p>Machine learning is a subset of artificial intelligence" in result
    
    def test_parse_paragraph_with_heading_success(self):
        """Test successful parsing of paragraph with heading."""
        content = "<h4>Understanding Machine Learning</h4>\n\n<p>Machine learning is a subset of AI.</p>"
        heading, paragraph = parse_paragraph_with_heading(content)
        
        assert heading == "Understanding Machine Learning"
        assert paragraph == "Machine learning is a subset of AI."
    
    def test_parse_paragraph_with_heading_failure(self):
        """Test fallback when parsing fails."""
        content = "Machine learning is a subset of AI."
        heading, paragraph = parse_paragraph_with_heading(content)
        
        assert heading == "Additional Information"
        assert paragraph == "Machine learning is a subset of AI."
    
    def test_convert_to_markdown_with_headings(self):
        """Test conversion to Markdown with headings."""
        html_content = """<h4>First Heading</h4>
        
        <p>First paragraph with <strong>bold</strong> and <em>italic</em> text.</p>
        
        <h4>Second Heading</h4>
        
        <p>Second paragraph content.</p>"""
        
        markdown = convert_to_markdown(html_content)
        
        assert "#### First Heading" in markdown
        assert "First paragraph with **bold** and *italic* text." in markdown
        assert "#### Second Heading" in markdown
        assert "Second paragraph content." in markdown
    
    def test_convert_to_markdown_without_headings(self):
        """Test conversion to Markdown without headings."""
        html_content = """<p>First paragraph with <strong>bold</strong> and <em>italic</em> text.</p>
        
        <p>Second paragraph content.</p>"""
        
        markdown = convert_to_markdown(html_content)
        
        assert "First paragraph with **bold** and *italic* text." in markdown
        assert "Second paragraph content." in markdown
    
    def test_parse_paragraph_with_heading_new_format(self):
        """Test successful parsing of paragraph with heading using new HTML-style tags."""
        content = "<paragraph>Machine learning is a subset of AI that enables computers to learn from data.</paragraph>\n<heading>Understanding Machine Learning</heading>"
        heading, paragraph = parse_paragraph_with_heading(content)
        
        assert heading == "Understanding Machine Learning"
        assert paragraph == "Machine learning is a subset of AI that enables computers to learn from data."
    
    def test_parse_paragraph_with_heading_legacy_format(self):
        """Test successful parsing of paragraph with heading using legacy square bracket format."""
        content = "[CONTENT] Machine learning is a subset of AI that enables computers to learn from data.\n[HEADING] Understanding Machine Learning"
        heading, paragraph = parse_paragraph_with_heading(content)
        
        assert heading == "Understanding Machine Learning"
        assert paragraph == "Machine learning is a subset of AI that enables computers to learn from data."
