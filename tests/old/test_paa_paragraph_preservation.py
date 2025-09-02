#!/usr/bin/env python3
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

"""
Test script to verify paragraph structure preservation in PAA content.

This script contains hard-coded PAA content in WordPress format and runs it through
both humanization and grammar correction processes to test if paragraph structures
are maintained correctly.
"""

import sys
import os
import re
from typing import Dict, Any, List

# Import necessary modules for mock implementation
from unittest.mock import MagicMock

# Create mock classes to avoid dependencies
class MockConfig:
    """Mock Config class for testing"""
    def __init__(self):
        self.enable_token_tracking = False
        self.grammar_check_temperature = 0.7
        self.humanization_temperature = 0.7
        self.enable_separate_grammar_model = False
        self.enable_separate_humanization_model = False
        self.openai_model = "gpt-3.5-turbo"
        self.add_paa_paragraphs_into_article = True
        self.serp_api_key = "dummy_key"

class MockPrompts:
    """Mock Prompts class for testing"""
    def __init__(self):
        pass
    
    def format_prompt(self, *args, **kwargs):
        return "Formatted prompt for testing"

class MockArticleContext:
    """Mock ArticleContext class for testing"""
    def __init__(self):
        self.config = MockConfig()
        self.prompts = MockPrompts()
        self.messages = []
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def count_message_tokens(self, message):
        return len(message.get("content", "")) // 4  # Rough approximation

# Mock the generate_completion function
def mock_generate_completion(*args, **kwargs):
    """Mock function to simulate AI response without making actual API calls"""
    prompt = kwargs.get('prompt', '')
    if not prompt:
        prompt = args[0] if args else ''
    
    # For humanization, simply return the input text with a marker
    if 'humanize this text' in prompt.lower():
        # Extract the text to be humanized
        text_to_process = prompt.split('TEXT TO HUMANIZE:')[-1].strip()
        # Simulate humanization by adding "humanized:" prefix to each paragraph
        paragraphs = text_to_process.split('\n\n')
        humanized_paragraphs = [f"Humanized: {p}" for p in paragraphs]
        return '\n\n'.join(humanized_paragraphs)
    
    # For grammar checking, return the input text with a marker
    elif 'check grammar' in prompt.lower():
        # Extract the text to check
        text_to_process = prompt.split('TEXT:')[-1].strip()
        # Simulate grammar checking by adding "grammar-fixed:" prefix to each paragraph
        paragraphs = text_to_process.split('\n\n')
        grammar_fixed_paragraphs = [f"Grammar-fixed: {p}" for p in paragraphs]
        return '\n\n'.join(grammar_fixed_paragraphs)
    
    # Default case
    return prompt

# Mock the actual modules with our mock implementations
import sys
sys.modules['script2.article_generator.text_processor'] = MagicMock()
sys.modules['script2.config'] = MagicMock()
sys.modules['script2.utils.prompts_config'] = MagicMock()
sys.modules['script2.article_generator.content_generator'] = MagicMock()
sys.modules['script2.utils.ai_utils'] = MagicMock()

# Mock the specific functions we need
sys.modules['script2.article_generator.text_processor'].humanize_text = MagicMock(side_effect=lambda *args, **kwargs: f"Humanized: {kwargs.get('text', args[1])}")
sys.modules['script2.article_generator.text_processor'].check_grammar = MagicMock(side_effect=lambda *args, **kwargs: f"Grammar-fixed: {kwargs.get('text', args[1])}")
sys.modules['script2.utils.ai_utils'].generate_completion = mock_generate_completion

# Create sample PAA content in WordPress format with multiple paragraphs per answer
SAMPLE_PAA_CONTENT = """# People Also Ask

## How do I grow tomatoes from seeds?

<p>First, you'll need to start with high-quality tomato seeds. Choose varieties that are well-suited to your growing zone and the space you have available. Start seeds indoors about 6-8 weeks before your last expected frost date. Use a seed-starting mix and plant seeds about 1/4 inch deep in seed trays or small pots.</p>

<p>After planting, keep the soil consistently moist but not waterlogged. Place the containers in a warm location (70-75°F is ideal) until germination occurs. Once seedlings emerge, move them to a location with plenty of light. A sunny window may work, but grow lights are often necessary to prevent seedlings from becoming leggy and weak.</p>

<p>When seedlings develop their first true leaves, thin them to one plant per cell or pot. About two weeks before transplanting outdoors, begin the process of hardening off by gradually exposing plants to outdoor conditions. Finally, transplant into garden soil that has been enriched with compost when all danger of frost has passed.</p>

## What is the best soil for growing tomatoes?

<p>The ideal soil for growing tomatoes is well-draining, fertile, and slightly acidic with a pH between 6.0 and 6.8. Tomatoes are heavy feeders and require nutrient-rich soil to produce healthy plants and abundant fruit. A good tomato soil should contain plenty of organic matter such as compost or well-rotted manure, which provides essential nutrients and helps maintain proper soil structure.</p>

<p>Before planting, consider amending your garden soil with a 2-3 inch layer of compost worked into the top 6-8 inches of soil. This improves fertility while enhancing the soil's ability to retain moisture while still allowing excess water to drain away. Many successful tomato growers also add specific amendments like bone meal for phosphorus or a balanced organic fertilizer formulated for tomatoes.</p>

## How often should I water tomato plants?

<p>Proper watering is critical for tomato plants, especially during fruit development. As a general rule, tomato plants need about 1-2 inches of water per week, though this can vary based on your climate, soil type, and the growth stage of the plants. In hot, dry conditions, you may need to water more frequently.</p>

<p>It's better to water deeply and less frequently rather than giving plants small amounts of water every day. Deep watering encourages roots to grow deeper into the soil, resulting in more resilient plants. Apply water at the base of the plant rather than overhead to keep foliage dry and reduce the risk of disease.</p>

<p>The best time to water tomato plants is in the early morning, allowing any moisture on leaves to dry during the day. Consistency is key—fluctuations in soil moisture can lead to problems like blossom end rot and fruit cracking. Using mulch around your plants helps maintain consistent soil moisture and reduces watering frequency.</p>

## What are common pests that affect tomato plants?

<p>Tomato plants can fall victim to various pests that can damage leaves, stems, and fruit. Some of the most common culprits include tomato hornworms, aphids, whiteflies, and spider mites. Tomato hornworms are large green caterpillars that can quickly defoliate plants, while aphids and whiteflies suck plant juices and can transmit diseases.</p>

<p>Another significant threat comes from soil-dwelling pests like cutworms, which can sever young plants at the soil line, and nematodes, which attack the root system. Fruit-specific pests include stink bugs and fruit worms, which can damage the tomatoes themselves, making them unsuitable for consumption.</p>

## How do I prevent tomato plant diseases?

<p>Prevention is the best approach for managing tomato diseases. Start by selecting disease-resistant varieties, which are labeled with letters like V (Verticillium wilt), F (Fusarium wilt), N (nematodes), and T (tobacco mosaic virus). Practice crop rotation, avoiding planting tomatoes or related plants in the same location for at least 3-4 years.</p>

<p>Proper spacing between plants improves air circulation and helps foliage dry quickly after rain or irrigation. Water at the base of plants rather than overhead to keep leaves dry, as many fungal and bacterial pathogens require moisture to infect plants. Apply mulch to prevent soil-borne diseases from splashing onto lower leaves during rainfall.</p>

<p>Maintain garden hygiene by removing and destroying diseased plant material promptly. Never compost diseased plants, as this can spread pathogens. At the end of the growing season, remove all tomato plant debris from the garden to eliminate overwintering sites for disease organisms.</p>
"""

# Create sample prompts for testing
SAMPLE_HUMANIZE_PROMPT = """You are an expert content humanizer. Please humanize this text to make it sound more natural and engaging. Do not change the meaning or structure of the content, just improve the language and flow.

TEXT TO HUMANIZE:
{humanize}

Make the humanized text flow naturally while preserving the original paragraph structure and meaning.
"""

SAMPLE_GRAMMAR_PROMPT = """You are an expert grammar checker. Please check and correct any grammar, spelling, or punctuation errors in the following text. Do not change the meaning or structure of the content, just fix any errors.

TEXT:
{text}

Return the corrected text with all grammar, spelling, and punctuation errors fixed.
"""

def create_mock_context():
    """Create a mock ArticleContext object for testing"""
    return MockArticleContext()

def process_paa_content_humanization(paa_content: str) -> str:
    """Process PAA content through humanization logic from generator.py"""
    # Parse PAA section to preserve structure - this is similar to code in generator.py
    paa_paragraphs = []
    current_paragraph = []
    is_header = False
    
    # Split by lines
    paa_lines = paa_content.split('\n')
    
    i = 0
    while i < len(paa_lines):
        line = paa_lines[i].rstrip()
        
        # Handle headings - preserve as-is
        if line.startswith('#'):
            # If we were building a paragraph, finalize it before starting new section
            if current_paragraph:
                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                current_paragraph = []
            
            current_paragraph.append(line)
            is_header = True
            
            # Add blank line after header if present
            if i + 1 < len(paa_lines) and not paa_lines[i + 1].strip():
                current_paragraph.append('')
                i += 1
        
        # Empty line - potential paragraph separator
        elif not line:
            if current_paragraph:
                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                current_paragraph = []
                is_header = False
            current_paragraph.append('')
        
        # Regular content line
        else:
            # If we were in a header and now we're not, finalize header paragraph
            if is_header and current_paragraph and not line.startswith('#'):
                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                current_paragraph = []
                is_header = False
            
            # Add this content line to current paragraph
            current_paragraph.append(line)
        
        i += 1
    
    # Don't forget the last paragraph if there is one
    if current_paragraph:
        paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
    
    # Create mock context for humanization
    context = create_mock_context()
    
    # Process each paragraph
    processed_paragraphs = []
    for paragraph_text, is_heading in paa_paragraphs:
        if is_heading or not paragraph_text.strip():
            # Skip processing for headings and empty lines
            processed_paragraphs.append(paragraph_text)
        else:
            # Process paragraph content
            try:
                # Process paragraph content with our mock function
                processed_paragraph = sys.modules['script2.article_generator.text_processor'].humanize_text(
                    context,
                    paragraph_text,
                    SAMPLE_HUMANIZE_PROMPT,
                    engine="gpt-3.5-turbo",
                    enable_token_tracking=False,
                    track_token_usage=False,
                    content_type="PAA Answer Paragraph"
                )
                processed_paragraphs.append(processed_paragraph)
            except Exception as e:
                print(f"Error humanizing PAA paragraph: {str(e)}")
                processed_paragraphs.append(paragraph_text)  # Keep original on error
    
    # Rebuild the PAA section with processed paragraphs
    return '\n'.join(processed_paragraphs)

def process_paa_content_grammar(paa_content: str) -> str:
    """Process PAA content through grammar checking logic"""
    # We'll use a similar approach as the humanization function
    paa_paragraphs = []
    current_paragraph = []
    is_header = False
    
    # Split by lines
    paa_lines = paa_content.split('\n')
    
    i = 0
    while i < len(paa_lines):
        line = paa_lines[i].rstrip()
        
        # Handle headings - preserve as-is
        if line.startswith('#'):
            # If we were building a paragraph, finalize it before starting new section
            if current_paragraph:
                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                current_paragraph = []
            
            current_paragraph.append(line)
            is_header = True
            
            # Add blank line after header if present
            if i + 1 < len(paa_lines) and not paa_lines[i + 1].strip():
                current_paragraph.append('')
                i += 1
        
        # Empty line - potential paragraph separator
        elif not line:
            if current_paragraph:
                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                current_paragraph = []
                is_header = False
            current_paragraph.append('')
        
        # Regular content line
        else:
            # If we were in a header and now we're not, finalize header paragraph
            if is_header and current_paragraph and not line.startswith('#'):
                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                current_paragraph = []
                is_header = False
            
            # Add this content line to current paragraph
            current_paragraph.append(line)
        
        i += 1
    
    # Don't forget the last paragraph if there is one
    if current_paragraph:
        paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
    
    # Create mock context for grammar checking
    context = create_mock_context()
    
    # Process each paragraph
    processed_paragraphs = []
    for paragraph_text, is_heading in paa_paragraphs:
        if is_heading or not paragraph_text.strip():
            # Skip processing for headings and empty lines
            processed_paragraphs.append(paragraph_text)
        else:
            # Process paragraph content
            try:
                # Process paragraph content with our mock function
                processed_paragraph = sys.modules['script2.article_generator.text_processor'].check_grammar(
                    context,
                    paragraph_text,
                    SAMPLE_GRAMMAR_PROMPT,
                    engine="gpt-3.5-turbo",
                    enable_token_tracking=False,
                    track_token_usage=False,
                    content_type="PAA Answer Paragraph"
                )
                processed_paragraphs.append(processed_paragraph)
            except Exception as e:
                print(f"Error checking grammar of PAA paragraph: {str(e)}")
                processed_paragraphs.append(paragraph_text)  # Keep original on error
    
    # Rebuild the PAA section with processed paragraphs
    return '\n'.join(processed_paragraphs)

def analyze_structure(content: str, label: str = "Content"):
    """Analyze the structure of PAA content to identify paragraph patterns"""
    print(f"\n=== {label} Structure Analysis ===")
    
    # Count main sections (questions)
    questions = re.findall(r'## [^\n]+', content)
    print(f"Number of questions: {len(questions)}")
    
    # Count paragraphs per answer
    sections = content.split('## ')[1:]  # Skip the first split which is the title
    for i, section in enumerate(sections):
        if not section.strip():
            continue
            
        lines = section.split('\n')
        question = lines[0]
        print(f"\nQuestion {i+1}: {question}")
        
        # Count <p> tags which indicate paragraphs
        paragraphs = re.findall(r'<p>.*?</p>', section, re.DOTALL)
        print(f"  Paragraphs in answer: {len(paragraphs)}")
        
        # Show each paragraph (truncated)
        for j, para in enumerate(paragraphs):
            # Clean the paragraph for display
            clean_para = re.sub(r'</?p>', '', para).strip()
            if len(clean_para) > 50:
                clean_para = clean_para[:50] + "..."
            print(f"  Paragraph {j+1}: {clean_para}")

def main():
    """Main test function"""
    print("=== PAA Paragraph Structure Preservation Test ===")
    print(f"Testing with sample PAA content ({len(SAMPLE_PAA_CONTENT)} characters)")
    
    # Analyze original structure
    analyze_structure(SAMPLE_PAA_CONTENT, "Original")
    
    # Test humanization
    print("\n\n=== Testing Humanization Process ===")
    humanized_content = process_paa_content_humanization(SAMPLE_PAA_CONTENT)
    print(f"Humanized content length: {len(humanized_content)} characters")
    analyze_structure(humanized_content, "After Humanization")
    
    # Test grammar checking
    print("\n\n=== Testing Grammar Checking Process ===")
    grammar_checked_content = process_paa_content_grammar(SAMPLE_PAA_CONTENT)
    print(f"Grammar-checked content length: {len(grammar_checked_content)} characters")
    analyze_structure(grammar_checked_content, "After Grammar Check")
    
    # Test both processes in sequence
    print("\n\n=== Testing Both Processes in Sequence ===")
    both_processed_content = process_paa_content_grammar(process_paa_content_humanization(SAMPLE_PAA_CONTENT))
    print(f"Fully processed content length: {len(both_processed_content)} characters")
    analyze_structure(both_processed_content, "After Both Processes")
    
    # Write results to files for detailed comparison
    with open("original_paa.txt", "w") as f:
        f.write(SAMPLE_PAA_CONTENT)
    with open("humanized_paa.txt", "w") as f:
        f.write(humanized_content)
    with open("grammar_checked_paa.txt", "w") as f:
        f.write(grammar_checked_content)
    with open("fully_processed_paa.txt", "w") as f:
        f.write(both_processed_content)
    
    print("\nTest results written to files for detailed comparison.")

if __name__ == "__main__":
    main()
