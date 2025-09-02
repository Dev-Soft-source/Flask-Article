#!/usr/bin/env python
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

"""
Test script to verify PAA multi-paragraph formatting fix
"""

import os
import sys
import importlib.util

# Add script2 to path
script2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'script2')
sys.path.append(script2_path)

# Import necessary modules
from config import Config
import prompts

def test_paa_prompt_formatting():
    """Test PAA prompt formatting with different paragraph counts"""
    print("Testing PAA prompt formatting with different paragraph counts...")
    
    # Test with different paragraph counts
    for paragraphs in [1, 2, 3]:
        # Calculate word count
        paragraphs_word_count = paragraphs * 100
        
        # Try formatting the prompt
        try:
            formatted_prompt = prompts.PAA_ANSWER_PROMPT.format(
                keyword="test keyword",
                question="test question",
                paragraphs=paragraphs,
                paragraphs_word_count=paragraphs_word_count,
                tone="neutral",
                language="English",
                audience="beginners",
                pov="second person"
            )
            print(f"✅ Successfully formatted PAA prompt with {paragraphs} paragraph(s)")
        except Exception as e:
            print(f"❌ Error formatting PAA prompt with {paragraphs} paragraph(s): {str(e)}")
            return False
    
    return True

if __name__ == "__main__":
    # Run the test
    success = test_paa_prompt_formatting()
    
    if success:
        print("\n✅ All tests passed! The PAA prompt formatting fix is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed! The PAA prompt formatting fix is not working correctly.")
        sys.exit(1)
