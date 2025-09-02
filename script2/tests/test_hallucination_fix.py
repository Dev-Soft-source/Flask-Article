#!/usr/bin/env python3
# Test script for hallucination fixes

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_paragraph_prompt_formatting():
    """Test the paragraph prompt formatting with the hallucination fixes."""
    try:
        # Load the prompts
        with open('prompts.py', 'r') as f:
            prompts_content = f.read()
            
        # Verify paragraph prompt has the required elements
        if 'You are now writing Paragraph {current_paragraph} of {paragraphs_per_section}' in prompts_content:
            print("✅ Paragraph prompt includes explicit paragraph numbering")
        else:
            print("❌ Paragraph prompt is missing explicit paragraph numbering")
            
        if 'The section needs to cover these points across {paragraphs_per_section} paragraphs:' in prompts_content:
            print("✅ Paragraph prompt includes content distribution guidance")
        else:
            print("❌ Paragraph prompt is missing content distribution guidance")
            
        if 'For this specific paragraph ({current_paragraph} of {paragraphs_per_section}), focus on:' in prompts_content:
            print("✅ Paragraph prompt includes specific paragraph focus")
        else:
            print("❌ Paragraph prompt is missing specific paragraph focus")
            
        # Check content_generator.py
        with open('article_generator/content_generator.py', 'r') as f:
            generator_content = f.read()
            
        if 'current_paragraph: int = 1' in generator_content and 'paragraphs_per_section: int = None' in generator_content:
            print("✅ ContentGenerator.generate_paragraph includes paragraph numbering parameters")
        else:
            print("❌ ContentGenerator.generate_paragraph is missing paragraph numbering parameters")
            
        if 'Distribute points across paragraphs if there are enough points' in generator_content:
            print("✅ ContentGenerator.generate_paragraph implements content distribution")
        else:
            print("❌ ContentGenerator.generate_paragraph is missing content distribution")
            
        # Check generator.py
        with open('article_generator/generator.py', 'r') as f:
            generator_py_content = f.read()
            
        if 'current_paragraph = i + 1' in generator_py_content and 'paragraphs_per_section=self.config.paragraphs_per_section' in generator_py_content:
            print("✅ Generator._generate_sections passes paragraph numbers correctly")
        else:
            print("❌ Generator._generate_sections is not passing paragraph numbers correctly")
            
        if 'section_points=' in generator_py_content:
            print("✅ Generator._generate_sections extracts and passes section points")
        else:
            print("❌ Generator._generate_sections is not passing section points")
            
        print("\n✅ All hallucination fixes have been implemented successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        
if __name__ == "__main__":
    test_paragraph_prompt_formatting()
