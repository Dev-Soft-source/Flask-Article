#!/usr/bin/env python3
# filepath: /home/abuh/Documents/Python/LLM_article_gen_2/scripts/script2/test_summary_keynotes_model.py

"""
Test script to verify that the summary and keynotes generation functions 
properly respect the enable_separate_summary_model flag and use the 
summary_keynotes_model when specified.
"""

import os
import sys
import json
from typing import Dict, Any
import argparse

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from article_generator.article_context import ArticleContext
from article_generator.text_processor import generate_block_notes
from article_generator.content_generator_summary import generate_article_summary
from article_generator.chunking_utils import combine_chunk_results_with_llm
from utils.rich_provider import provider
from utils.ai_utils import generate_completion, make_openrouter_api_call

# Import configuration
from main import get_config

def create_test_article() -> Dict[str, Any]:
    """Create a sample article for testing purposes."""
    return {
        "title": "Test Article for Summary and Keynotes Generation",
        "introduction": "This is a test introduction paragraph. It contains some information about the topic.",
        "sections": [
            "This is the first section of the article. It contains information about the first aspect of the topic.",
            "This is the second section of the article. It contains information about the second aspect of the topic."
        ],
        "conclusion": "This is the conclusion of the article. It summarizes the main points discussed.",
        "keyword": "test article"
    }

def test_summary_keynotes_model_selection():
    """Test that the summary and keynotes model selection is working correctly."""
    provider.info("Starting test of summary and keynotes model selection")
    
    # Get configuration with separate summary model enabled
    config = get_config()
    
    # Ensure that the separate summary model flag is enabled
    config.enable_separate_summary_model = True
    config.summary_keynotes_model = "test-model-for-summary-keynotes"
    
    # Create context with this configuration
    context = ArticleContext(config)
    
    # Create test article
    article = create_test_article()
    
    # Test prompts
    summarize_prompt = "Create a summary of the following article:\n\n{article_content}"
    combine_prompt = "Combine these chunks into a single coherent summary:\n\n{chunks_text}"
    blocknote_prompt = "Create key takeaways for the following article:\n\n{article_content}"
    
    # Override the generate_completion function to track model usage
    original_generate_completion = generate_completion
    model_used_list = []
    
    def mock_generate_completion(prompt, model, **kwargs):
        """Mock function to track model usage."""
        model_used_list.append(model)
        provider.info(f"Model used: {model}")
        return f"Test completion using model: {model}"
    
    # Override the make_openrouter_api_call function
    original_make_openrouter_api_call = make_openrouter_api_call
    
    def mock_make_openrouter_api_call(messages, model, **kwargs):
        """Mock function to track OpenRouter model usage."""
        model_used_list.append(model)
        provider.info(f"OpenRouter model used: {model}")
        
        class MockResponse:
            class Choice:
                class Message:
                    content = f"Test completion using OpenRouter model: {model}"
                
                message = Message()
            
            choices = [Choice()]
        
        return MockResponse()
    
    # Apply the mocks
    sys.modules["utils.ai_utils"].generate_completion = mock_generate_completion
    sys.modules["utils.ai_utils"].make_openrouter_api_call = mock_make_openrouter_api_call
    
    try:
        # Test blocknotes generation
        provider.info("Testing block notes generation...")
        block_notes = generate_block_notes(
            context=context,
            article_content=article,
            blocknote_prompt=blocknote_prompt,
            combine_prompt=combine_prompt,
            engine="default-engine"
        )
        
        # Test summary generation
        provider.info("Testing summary generation...")
        summary = generate_article_summary(
            context=context,
            keyword=article["keyword"],
            article_dict=article,
            summarize_prompt=summarize_prompt,
            combine_prompt=combine_prompt
        )
        
        # Test combining chunks
        provider.info("Testing chunk combination...")
        combined = combine_chunk_results_with_llm(
            results=["Chunk 1 result", "Chunk 2 result"],
            context=context,
            combine_prompt=combine_prompt,
            is_summary=True
        )
        
        # Check if all model usages are correct
        all_correct = all(model == config.summary_keynotes_model for model in model_used_list)
        
        if all_correct:
            provider.success("✓ All tests passed! The summary_keynotes_model was properly used for all operations.")
        else:
            provider.error("✗ Test failed! Not all operations used the summary_keynotes_model.")
            for i, model in enumerate(model_used_list):
                provider.info(f"Model {i+1}: {model}")
    
    finally:
        # Restore original functions
        sys.modules["utils.ai_utils"].generate_completion = original_generate_completion
        sys.modules["utils.ai_utils"].make_openrouter_api_call = original_make_openrouter_api_call

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test summary and keynotes model selection")
    parser.add_argument("--with-openrouter", action="store_true", help="Test with OpenRouter enabled")
    args = parser.parse_args()
    
    if args.with_openrouter:
        provider.info("Testing with OpenRouter enabled")
        # Get config and enable OpenRouter
        config = get_config()
        config.use_openrouter = True
        config.openrouter_api_key = "test-api-key"
    
    test_summary_keynotes_model_selection()
