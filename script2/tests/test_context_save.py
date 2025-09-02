# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ
# Test script for context saving

import os
import sys
from config import Config
from utils.prompts_config import Prompts
from article_generator.article_context import ArticleContext
from utils.ai_utils import generate_completion
from utils.rich_provider import provider

def main():
    """Test context saving functionality."""
    try:
        provider.info("Testing context saving functionality...")

        # Create a test config with context saving enabled
        config = Config(
            # Enable context saving
            enable_context_save=True,
            context_save_dir="article_contexts_test",

            # API settings
            openai_model="gpt-4o-mini-2024-07-18",
            use_openrouter=True,
            openrouter_model="meta-llama/llama-3.3-70b-instruct:free",

            # Other required settings
            articlelanguage="English",
            articleaudience="General",
            articletype="Default",
            voicetone="friendly",
            pointofview="Third Person",
            sizesections=3,
            sizeheadings=2,
            paragraphs_per_section=2,

            # Token settings
            max_context_window_tokens=128000,
            token_padding=1000,
            warn_token_threshold=0.9,
            enable_token_tracking=True,
        )

        # Create prompts
        prompts = Prompts(
            title="Test title prompt",
            outline="Test outline prompt",
            introduction="Test introduction prompt",
            paragraph="Test paragraph prompt",
            conclusion="Test conclusion prompt",
            system_message="Test system message",
            faq="Test FAQ prompt",
            meta_description="Test meta description prompt",
            wordpress_excerpt="Test WordPress excerpt prompt",
            grammar="Test grammar prompt",
            humanize="Test humanize prompt",
            blocknote="Test blocknote prompt",
            summarize="Test summarize prompt",
            paa_answer="Test PAA answer prompt"
        )

        # Create article context
        context = ArticleContext(config, prompts)

        # Add some test messages manually
        context.add_message("user", "This is a test user message 1")
        context.add_message("assistant", "This is a test assistant response 1")

        # Test generate_completion function to see if it adds messages to context
        provider.info("Testing generate_completion function...")
        test_prompt = "Write a short paragraph about artificial intelligence."

        # Generate completion should add both the prompt and response to context
        # Determine which model to use based on whether OpenRouter is enabled
        model_to_use = config.openrouter_model if (hasattr(config, 'use_openrouter') and config.use_openrouter and config.openrouter_api_key) else config.openai_model
        
        response = generate_completion(
            prompt=test_prompt,
            model=model_to_use,
            temperature=0.7,
            max_tokens=100,
            article_context=context
        )

        provider.info(f"Generated response: {response[:50]}...")

        # Add some article parts
        context.article_parts["title"] = "Test Title"
        context.article_parts["outline"] = "Test Outline"
        context.article_parts["introduction"] = "Test Introduction"
        context.article_parts["sections"] = ["Test Section 1", "Test Section 2"]
        context.article_parts["conclusion"] = "Test Conclusion"

        # Save context to file
        filepath = context.save_to_file("test_context.md")

        if filepath:
            provider.success(f"Context saved to {filepath}")
            provider.info(f"Number of messages in context: {len(context.messages)}")

            # Print message roles to verify
            message_roles = [msg["role"] for msg in context.messages]
            provider.info(f"Message roles: {message_roles}")

            # Print the content of the saved file
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    provider.info("\nFile content:")
                    print("=" * 50)
                    print(f.read())
                    print("=" * 50)
        else:
            provider.error("Failed to save context")

    except Exception as e:
        provider.error(f"Error in test: {str(e)}")
        import traceback
        provider.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
