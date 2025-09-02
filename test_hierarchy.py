#!/usr/bin/env python3
"""
Test script to verify hierarchical structure handling in article generation.
Tests both 2-level (sections → subsections) and 3-level (sections → subsections → paragraph points) hierarchies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'script1'))

from article_generator.content_generator import parse_outline

def test_2level_hierarchy():
    """Test 2-level hierarchy parsing."""
    print("Testing 2-level hierarchy...")
    
    # Sample 2-level outline
    outline_2level = """I. Introduction to Digital Marketing
A. Understanding the digital landscape
B. Key marketing channels
C. Setting goals and objectives
D. Budget allocation strategies
E. Measuring success metrics

II. Content Marketing Strategies
A. Creating valuable content
B. Content distribution channels
C. SEO optimization techniques
D. Social media integration
E. Performance tracking"""

    # Parse the outline
    sections = parse_outline(outline_2level)
    
    print(f"Parsed {len(sections)} sections:")
    for section in sections:
        print(f"  Section: {section['title']}")
        print(f"  Has paragraph points: {section.get('has_paragraph_points', False)}")
        print(f"  Subsections: {len(section['subsections'])}")
        for i, subsection in enumerate(section['subsections'], 1):
            print(f"    {i}. {subsection['title']}")
        print()

def test_3level_hierarchy():
    """Test 3-level hierarchy parsing."""
    print("Testing 3-level hierarchy...")
    
    # Sample 3-level outline
    outline_3level = """I. Introduction to Digital Marketing
A. Understanding the digital landscape
   1. Evolution from traditional to digital
   2. Current market trends
   3. Consumer behavior shifts
B. Key marketing channels
   1. Search engine marketing
   2. Social media platforms
   3. Email marketing
   4. Content marketing
C. Setting goals and objectives
   1. SMART goal framework
   2. KPI selection
   3. Timeline planning

II. Content Marketing Strategies
A. Creating valuable content
   1. Understanding audience needs
   2. Content ideation techniques
   3. Quality vs quantity balance
B. Content distribution channels
   1. Owned media platforms
   2. Earned media opportunities
   3. Paid media strategies"""

    # Parse the outline
    sections = parse_outline(outline_3level)
    
    print(f"Parsed {len(sections)} sections:")
    for section in sections:
        print(f"  Section: {section['title']}")
        print(f"  Has paragraph points: {section.get('has_paragraph_points', False)}")
        print(f"  Subsections: {len(section['subsections'])}")
        for i, subsection in enumerate(section['subsections'], 1):
            print(f"    {i}. {subsection['title']}")
            if 'paragraph_points' in subsection:
                for j, point in enumerate(subsection['paragraph_points'], 1):
                    print(f"      {j}. {point}")
        print()

if __name__ == "__main__":
    print("=== Hierarchical Structure Test ===\n")
    
    test_2level_hierarchy()
    print("\n" + "="*50 + "\n")
    test_3level_hierarchy()
    
    print("=== Test Complete ===")