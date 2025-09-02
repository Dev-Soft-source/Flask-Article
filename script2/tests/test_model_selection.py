#!/usr/bin/env python
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

"""
Test script for validating the model selection functionality for grammar checking and humanization.
This test creates a small ArticleContext and tests using different models for grammar and humanization.
"""

import tiktoken
from article_generator.content_generator import ArticleContext
from article_generator.text_processor import humanize_text, check_grammar
from config import Config
from utils.rich_provider import provider

def main():
    # Initialize the configuration with test settings
    config = Config(
        # Enable OpenRouter
        use_openrouter=True,
        
        # Main model for content generation
        openrouter_model="meta-llama/llama-3-70b-instruct",
        
        # Enable separate models for grammar and humanization
        enable_separate_grammar_model=True,
        grammar_check_model="anthropic/claude-3-haiku-20240307",
        
        enable_separate_humanization_model=True,
        humanization_model="anthropic/claude-3-sonnet-20240229",
        
        # Temperature settings
        grammar_check_temperature=0.3,
        humanization_temperature=0.7,
    )
    
    # Create a simple article context
    context = ArticleContext(
        keyword="test keyword",
        config=config,
        encoding=tiktoken.encoding_for_model("gpt-4")
    )
    
    # Test text for grammar check and humanization
    test_text = "This sentence have grammar errors. The writer don't know how to writes well."
    
    # Grammar prompt
    grammar_prompt = """
    You are a professional grammar checker and editor. Review the following text and correct any grammar, 
    spelling, or punctuation errors. Keep the meaning and style of the original text intact.
    
    Text: {text}
    
    Corrected text:
    """
    
    # Humanization prompt
    humanize_prompt = """
    You are a professional writer who specializes in making text sound more human and natural. 
    Please rewrite the following text to make it more engaging, conversational, and human-like 
    while preserving the original meaning.
    
    Text: {humanize}
    
    Humanized text:
    """
    
    provider.info("Testing grammar check with separate model...")
    provider.info(f"Original text: {test_text}")
    
    # First, test with separate grammar model
    corrected_text = check_grammar(
        context=context,
        text=test_text,
        grammar_prompt=grammar_prompt,
        engine=config.openai_model,
        content_type="test content"
    )
    provider.info(f"Grammar-corrected text (using {config.grammar_check_model}): {corrected_text}")
    
    provider.info("\nTesting humanization with separate model...")
    
    # Test with separate humanization model
    humanized_text = humanize_text(
        context=context,
        text=test_text,
        humanize_prompt=humanize_prompt,
        engine=config.openai_model,
        content_type="test content"
    )
    provider.info(f"Humanized text (using {config.humanization_model}): {humanized_text}")
    
    # Now test with same model for all tasks
    provider.info("\nTesting with single model for all tasks...")
    
    # Disable separate models
    config.enable_separate_grammar_model = False
    config.enable_separate_humanization_model = False
    
    corrected_text = check_grammar(
        context=context,
        text=test_text,
        grammar_prompt=grammar_prompt,
        engine=config.openai_model,
        content_type="test content"
    )
    provider.info(f"Grammar-corrected text (using main model {config.openrouter_model}): {corrected_text}")
    
    humanized_text = humanize_text(
        context=context,
        text=test_text,
        humanize_prompt=humanize_prompt,
        engine=config.openai_model,
        content_type="test content"
    )
    provider.info(f"Humanized text (using main model {config.openrouter_model}): {humanized_text}")
    
if __name__ == "__main__":
    main()
