#!/usr/bin/env python3
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

"""
Test script for verifying grammar check functionality in Script 2
This script tests the fixes for rate limiting issues with OpenRouter.
"""

import os
import sys
import time
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).resolve().parent))

from config import Config
from article_generator.content_generator import ArticleContext
from article_generator.text_processor import check_grammar
from utils.rich_provider import provider
import tiktoken

# Sample text with intentional grammar errors
SAMPLE_TEXT = """
This article explore the impact of artifical intellegence on modern society. 
AI have been increasingly prevalent in our daily lifes. Many company are 
implementing AI solutions to improve they're operations and customer experience.
Despite it's benefits, their are concerns about privacy, job displacement, and 
ethical implications what need to be addressed. The technology advancing rapidly, 
but regulatory framework is not keeping pace with this advancements.
"""

# Grammar checking prompt
GRAMMAR_PROMPT = """
You are a professional grammar and language specialist. Please carefully review the following text and fix any grammatical errors, spelling mistakes, punctuation issues, and improve readability without changing the meaning or style of the content.

Keep all formatting (such as markdown, bullet points, etc.) intact. Only fix language errors.

Text to check:
{text}

Corrected text:
"""

def main():
    """Main function to test grammar checking functionality"""
    # Initialize configuration
    config = Config()
    
    # Enable grammar checking
    config.enable_grammar_check = True
    
    # Configure OpenRouter
    config.use_openrouter = True
    config.openrouter_model = "meta-llama/llama-3-70b-instruct:free"  # Using free tier model to test rate limiting
    
    # Ensure API key is set
    if not config.openrouter_api_key and os.environ.get("OPENROUTER_API_KEY"):
        config.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    
    if not config.openrouter_api_key:
        provider.error("OpenRouter API key not found. Please set the OPENROUTER_API_KEY environment variable.")
        return
    
    # Initialize tokenizer for tracking
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Create ArticleContext
    context = ArticleContext(
        keyword="grammar testing",
        config=config,
        encoding=encoding
    )
    
    # Display test information
    provider.info("Testing grammar checking functionality")
    provider.info(f"OpenRouter model: {config.openrouter_model}")
    provider.info(f"Original text length: {len(SAMPLE_TEXT)}")
    
    # Print original text
    provider.info("Original text (with errors):")
    print(f"\n{SAMPLE_TEXT}\n")
    
    # Test grammar checking
    try:
        provider.info("Running grammar check...")
        corrected_text = check_grammar(
            context=context,
            text=SAMPLE_TEXT,
            grammar_prompt=GRAMMAR_PROMPT,
            engine=config.openrouter_model,
            enable_token_tracking=True,
            track_token_usage=True,
            content_type="test content"
        )
        
        # Print corrected text
        provider.success("Grammar check completed successfully!")
        provider.info("Corrected text:")
        print(f"\n{corrected_text}\n")
        
        # Run again to test rate limiting handling
        provider.info("Running grammar check again immediately to test rate limiting...")
        time.sleep(1)  # Small pause
        
        corrected_text_2 = check_grammar(
            context=context,
            text=SAMPLE_TEXT,
            grammar_prompt=GRAMMAR_PROMPT,
            engine=config.openrouter_model,
            enable_token_tracking=True,
            track_token_usage=True,
            content_type="test content (second run)"
        )
        
        # Print second corrected text
        provider.success("Second grammar check completed!")
        provider.info("Second corrected text:")
        print(f"\n{corrected_text_2}\n")
        
    except Exception as e:
        provider.error(f"Error during grammar check: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
