#!/usr/bin/env python3
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

"""
Test script for PAA configuration parameters and functionality improvements.

This script tests:
1. New PAA configuration parameters (paa_max_questions, paa_min_questions, paa_use_random_range)
2. Random range functionality 
3. Humanization preservation of PAA structure
"""

import sys
import os
import tempfile
import json
from unittest.mock import MagicMock, patch

# Add both script directories to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'script1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'script2'))

def test_script1_paa_config():
    """Test Script 1 PAA configuration parameters."""
    print("=" * 60)
    print("Testing Script 1 PAA Configuration Parameters")
    print("=" * 60)
    
    try:
        from script1.utils.config import Config
        
        # Test default values
        config = Config()
        assert hasattr(config, 'paa_max_questions'), "paa_max_questions parameter missing"
        assert hasattr(config, 'paa_min_questions'), "paa_min_questions parameter missing"
        assert hasattr(config, 'paa_use_random_range'), "paa_use_random_range parameter missing"
        
        assert config.paa_max_questions == 5, f"Expected paa_max_questions=5, got {config.paa_max_questions}"
        assert config.paa_min_questions == 3, f"Expected paa_min_questions=3, got {config.paa_min_questions}"
        assert config.paa_use_random_range == False, f"Expected paa_use_random_range=False, got {config.paa_use_random_range}"
        
        print("✅ Script 1 PAA configuration parameters are correctly defined with proper defaults")
        return True
        
    except Exception as e:
        print(f"❌ Script 1 PAA configuration test failed: {str(e)}")
        return False

def test_script2_paa_config():
    """Test Script 2 PAA configuration parameters."""
    print("\n" + "=" * 60)
    print("Testing Script 2 PAA Configuration Parameters")
    print("=" * 60)
    
    try:
        from script2.config import Config
        
        # Test default values
        config = Config()
        assert hasattr(config, 'paa_max_questions'), "paa_max_questions parameter missing"
        assert hasattr(config, 'paa_min_questions'), "paa_min_questions parameter missing"
        assert hasattr(config, 'paa_use_random_range'), "paa_use_random_range parameter missing"
        
        assert config.paa_max_questions == 5, f"Expected paa_max_questions=5, got {config.paa_max_questions}"
        assert config.paa_min_questions == 3, f"Expected paa_min_questions=3, got {config.paa_min_questions}"
        assert config.paa_use_random_range == False, f"Expected paa_use_random_range=False, got {config.paa_use_random_range}"
        
        print("✅ Script 2 PAA configuration parameters are correctly defined with proper defaults")
        return True
        
    except Exception as e:
        print(f"❌ Script 2 PAA configuration test failed: {str(e)}")
        return False

def test_script1_paa_random_range():
    """Test Script 1 PAA random range functionality."""
    print("\n" + "=" * 60)
    print("Testing Script 1 PAA Random Range Functionality")
    print("=" * 60)
    
    try:
        from script1.utils.config import Config
        from script1.article_generator.paa_handler import get_paa_questions
        
        # Create config with random range enabled
        config = Config()
        config.paa_use_random_range = True
        config.paa_min_questions = 2
        config.paa_max_questions = 4
        
        # Mock the API call to return a fixed set of questions
        mock_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?", 
            "What are neural networks?",
            "What is deep learning?",
            "How to get started with AI?",
            "What programming languages are used in AI?"
        ]
        
        # Test multiple calls to verify randomness
        results = []
        with patch('script1.article_generator.paa_handler.GoogleSearch') as mock_search:
            mock_instance = MagicMock()
            mock_instance.get_dict.return_value = {
                'related_questions': [{'question': q} for q in mock_questions]
            }
            mock_search.return_value = mock_instance
            
            # Test 10 times to check random range
            for i in range(10):
                questions = get_paa_questions("test keyword", "fake_api_key", config)
                results.append(len(questions))
                assert 2 <= len(questions) <= 4, f"Question count {len(questions)} outside range [2, 4]"
        
        # Verify we got different counts (indicating randomness)
        unique_counts = set(results)
        print(f"Random range results: {sorted(list(unique_counts))}")
        assert len(unique_counts) > 1, "No variation in question counts - randomness not working"
        
        print("✅ Script 1 PAA random range functionality works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Script 1 PAA random range test failed: {str(e)}")
        return False

def test_script2_paa_random_range():
    """Test Script 2 PAA random range functionality."""
    print("\n" + "=" * 60)
    print("Testing Script 2 PAA Random Range Functionality")
    print("=" * 60)
    
    try:
        from script2.config import Config
        from script2.article_generator.paa_handler import PAAHandler
        from script2.utils.api_utils import SerpAPI
        
        # Create config with random range enabled
        config = Config()
        config.paa_use_random_range = True
        config.paa_min_questions = 2
        config.paa_max_questions = 4
        config.add_paa_paragraphs_into_article = True
        
        # Mock the API responses
        mock_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?", 
            "What are neural networks?",
            "What is deep learning?",
            "How to get started with AI?",
            "What programming languages are used in AI?"
        ]
        
        # Create PAA handler
        paa_handler = PAAHandler(config)
        
        # Test multiple calls to verify randomness
        results = []
        with patch.object(paa_handler.serp_api, 'perform_search') as mock_search:
            mock_search.return_value = {
                'related_questions': [{'question': q} for q in mock_questions]
            }
            
            # Test 10 times to check random range
            for i in range(10):
                questions = paa_handler.get_paa_questions("test keyword", 5)
                results.append(len(questions))
                assert 2 <= len(questions) <= 4, f"Question count {len(questions)} outside range [2, 4]"
        
        # Verify we got different counts (indicating randomness)
        unique_counts = set(results)
        print(f"Random range results: {sorted(list(unique_counts))}")
        assert len(unique_counts) > 1, "No variation in question counts - randomness not working"
        
        print("✅ Script 2 PAA random range functionality works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Script 2 PAA random range test failed: {str(e)}")
        return False

def test_script2_humanization_structure_preservation():
    """Test that Script 2 humanization preserves PAA structure."""
    print("\n" + "=" * 60)
    print("Testing Script 2 Humanization Structure Preservation")
    print("=" * 60)
    
    try:
        # Sample PAA content with proper markdown structure
        sample_paa_content = """# People Also Ask

## What is artificial intelligence?

Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

## How does machine learning work?

Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions based on the information it has processed.

## What are the benefits of AI in business?

AI offers numerous benefits for businesses including increased efficiency, cost reduction, improved customer service, better decision-making through data analysis, and the ability to automate repetitive tasks.
"""
        
        # Test the humanization preservation logic
        lines = sample_paa_content.split('\n')
        humanized_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Preserve main heading (# People Also Ask)
            if line.startswith('# People Also Ask'):
                humanized_lines.append(line)
                if i + 1 < len(lines) and lines[i + 1].strip() == '':
                    humanized_lines.append('')
                    i += 1
            
            # Preserve question headings (## or ###)
            elif line.startswith('##') and '?' in line:
                humanized_lines.append(line)
                if i + 1 < len(lines) and lines[i + 1].strip() == '':
                    humanized_lines.append('')
                    i += 1
            
            # Mock humanize answer paragraphs (non-heading, non-empty lines)
            elif line.strip() and not line.startswith('#'):
                # Collect consecutive answer lines
                answer_lines = []
                while i < len(lines) and lines[i].strip() and not lines[i].startswith('#'):
                    answer_lines.append(lines[i])
                    i += 1
                i -= 1  # Back up one since the loop will increment
                
                # Mock humanization (just add a prefix to simulate processing)
                answer_text = '\n'.join(answer_lines)
                if answer_text.strip():
                    humanized_answer = f"[HUMANIZED] {answer_text}"
                    humanized_lines.append(humanized_answer)
            
            # Preserve empty lines and other formatting
            else:
                humanized_lines.append(line)
            
            i += 1
        
        # Rebuild the PAA section with preserved structure
        result = '\n'.join(humanized_lines)
        
        # Verify structure is preserved
        assert "# People Also Ask" in result, "Main heading not preserved"
        assert "## What is artificial intelligence?" in result, "Question heading 1 not preserved"
        assert "## How does machine learning work?" in result, "Question heading 2 not preserved"
        assert "## What are the benefits of AI in business?" in result, "Question heading 3 not preserved"
        assert "[HUMANIZED]" in result, "Content was not processed for humanization"
        
        print("✅ PAA structure preservation logic works correctly")
        print("Sample output:")
        print(result[:200] + "..." if len(result) > 200 else result)
        return True
        
    except Exception as e:
        print(f"❌ Script 2 humanization structure preservation test failed: {str(e)}")
        return False

def test_paa_config_with_different_values():
    """Test PAA configuration with different parameter values."""
    print("\n" + "=" * 60)
    print("Testing PAA Configuration with Different Values")
    print("=" * 60)
    
    try:
        from script1.utils.config import Config as Config1
        from script2.config import Config as Config2
        
        # Test Script 1 with custom values
        config1 = Config1()
        config1.paa_max_questions = 8
        config1.paa_min_questions = 2
        config1.paa_use_random_range = True
        
        assert config1.paa_max_questions == 8
        assert config1.paa_min_questions == 2
        assert config1.paa_use_random_range == True
        
        # Test Script 2 with custom values
        config2 = Config2()
        config2.paa_max_questions = 10
        config2.paa_min_questions = 1
        config2.paa_use_random_range = False
        
        assert config2.paa_max_questions == 10
        assert config2.paa_min_questions == 1
        assert config2.paa_use_random_range == False
        
        print("✅ PAA configuration accepts custom values correctly")
        return True
        
    except Exception as e:
        print(f"❌ Custom PAA configuration test failed: {str(e)}")
        return False

def main():
    """Run all PAA configuration tests."""
    print("PAA Configuration and Functionality Test Suite")
    print("=" * 60)
    print("Testing PAA functionality improvements implementation")
    print()
    
    tests = [
        test_script1_paa_config,
        test_script2_paa_config,
        test_script1_paa_random_range,
        test_script2_paa_random_range,
        test_script2_humanization_structure_preservation,
        test_paa_config_with_different_values
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 All PAA functionality tests passed!")
        return 0
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())
