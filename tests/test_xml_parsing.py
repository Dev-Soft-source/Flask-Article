#!/usr/bin/env python3
"""
Test script to verify XML tag parsing functionality
"""

import re

def extract_content_from_xml_tags(response_text: str, prompt: str) -> str:
    """
    Extract content from XML tags based on tags mentioned in the prompt.
    If no tags are found or extraction fails, return the original response.
    """
    # Find XML tags mentioned in the prompt
    xml_tag_pattern = r'<([a-zA-Z][a-zA-Z0-9_]*)>'
    mentioned_tags = re.findall(xml_tag_pattern, prompt)
    
    if not mentioned_tags:
        # No XML tags found in prompt, return original response
        return response_text
    
    # Get the most likely tag (usually the last one mentioned)
    target_tag = mentioned_tags[-1]
    
    # Create regex pattern to extract content from the target tag
    content_pattern = fr'<{target_tag}[^>]*>(.*?)</{target_tag}>'
    
    # Search for the content within the target tag
    match = re.search(content_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        # Successfully extracted content from the tag
        extracted_content = match.group(1).strip()
        print(f"✓ Extracted content from <{target_tag}> tags")
        return extracted_content
    else:
        # Tag not found in response, return original response as fallback
        print(f"⚠ Expected <{target_tag}> tags not found, using full response")
        return response_text

# Test cases
def test_parsing():
    print("Testing XML tag parsing...")
    
    # Test 1: Successful extraction
    prompt1 = "Write a title in <title> tags"
    response1 = "<title>How to Bake Perfect Bread</title>"
    result1 = extract_content_from_xml_tags(response1, prompt1)
    print(f"Test 1 - Expected: 'How to Bake Perfect Bread', Got: '{result1}'")
    assert result1 == "How to Bake Perfect Bread"
    
    # Test 2: No tags in prompt
    prompt2 = "Write a simple paragraph"
    response2 = "This is a simple paragraph without tags."
    result2 = extract_content_from_xml_tags(response2, prompt2)
    print(f"Test 2 - Expected: 'This is a simple paragraph without tags.', Got: '{result2}'")
    assert result2 == "This is a simple paragraph without tags."
    
    # Test 3: Tags not found in response (fallback)
    prompt3 = "Write content in <outline> tags"
    response3 = "I. Introduction\nII. Main Content\nIII. Conclusion"
    result3 = extract_content_from_xml_tags(response3, prompt3)
    print(f"Test 3 - Expected: 'I. Introduction\\nII. Main Content\\nIII. Conclusion', Got: '{result3}'")
    assert result3 == "I. Introduction\nII. Main Content\nIII. Conclusion"
    
    # Test 4: Multiple tags in prompt - should extract from the last mentioned tag
    prompt4 = "Return in <paragraph> tags with <heading>"
    response4 = "<paragraph>This is the content</paragraph><heading>Section Title</heading>"
    result4 = extract_content_from_xml_tags(response4, prompt4)
    print(f"Test 4 - Expected: 'Section Title', Got: '{result4}'")
    assert result4 == "Section Title"
    
    # Test 5: Multi-line content
    prompt5 = "Write in <intro> tags"
    response5 = "<intro>\nWelcome to our guide!\nThis will help you learn everything.\n</intro>"
    result5 = extract_content_from_xml_tags(response5, prompt5)
    print(f"Test 5 - Expected: '\\nWelcome to our guide!\\nThis will help you learn everything.', Got: '{repr(result5)}'")
    assert "Welcome to our guide!" in result5
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_parsing()