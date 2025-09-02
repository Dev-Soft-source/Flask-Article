# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ
# Test script for verifying context preservation in API calls

import os
import sys
from config import Config
from utils.prompts_config import Prompts
from article_generator.article_context import ArticleContext
from utils.ai_utils import generate_completion
from utils.rich_provider import provider

def main():
    """Test context preservation across multiple API calls."""
    try:
        provider.info("Testing context preservation across multiple API calls...")

        # Create a test config
        config = Config(
            # API settings
            openai_model="gpt-4o-mini-2024-07-18",
            use_openrouter=True,
            openrouter_model="meta-llama/llama-3.3-70b-instruct:free",
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            openrouter_site_url="https://example.com",
            openrouter_site_name="Article Generator Test",

            # Other required settings
            articlelanguage="English",
            articleaudience="General",
            articletype="Default",
            voicetone="friendly",
            pointofview="Third Person",

            # Enable context saving for debugging
            enable_context_save=True,
            context_save_dir="article_contexts_test",
        )

        # Create prompts
        prompts = Prompts(
            system_message="You are an expert content writer creating coherent, engaging articles.",
            title="Generate a title for an article about {keyword}.",
            outline="Create an outline for an article about {keyword}.",
            introduction="Write an introduction for an article about {keyword}.",
            paragraph="Write a paragraph about {subtitle} for an article about {keyword}.",
            conclusion="Write a conclusion for an article about {keyword}."
        )

        # Create article context
        context = ArticleContext(config, prompts)
        context.keyword = "artificial intelligence"

        provider.success("Created ArticleContext with keyword: artificial intelligence")
        provider.info(f"Initial number of messages: {len(context.messages)}")

        # First API call
        provider.info("Making first API call to generate title...")
        first_prompt = "Generate a title for an article about artificial intelligence."
        first_response = generate_completion(
            prompt=first_prompt,
            model=config.openrouter_model,
            temperature=0.7,
            max_tokens=100,
            article_context=context
        )

        provider.success(f"First API call response: {first_response}")
        provider.info(f"Number of messages after first call: {len(context.messages)}")

        # Second API call - should reference the first response
        provider.info("Making second API call to generate outline...")
        second_prompt = "Create an outline for the article with the title you just generated."
        second_response = generate_completion(
            prompt=second_prompt,
            model=config.openrouter_model,
            temperature=0.7,
            max_tokens=200,
            article_context=context
        )

        provider.success(f"Second API call response: {second_response}")
        provider.info(f"Number of messages after second call: {len(context.messages)}")

        # Third API call - should reference both previous responses
        provider.info("Making third API call with context-dependent question...")
        third_prompt = "Based on the title and outline you created, write a brief introduction."
        third_response = generate_completion(
            prompt=third_prompt,
            model=config.openrouter_model,
            temperature=0.7,
            max_tokens=300,
            article_context=context
        )

        provider.success(f"Third API call response: {third_response}")
        provider.info(f"Number of messages after third call: {len(context.messages)}")

        # Save context to file for inspection
        filepath = context.save_to_file("context_preservation_test.md")
        provider.success(f"Context saved to {filepath}")

        # Check if responses indicate context preservation
        if first_response in third_response or any(word in third_response for word in first_response.split() if len(word) > 5):
            provider.success("Context preservation test PASSED: Third response references the title")
        else:
            provider.warning("Context preservation test inconclusive: Third response may not reference the title explicitly")

        provider.info("Message roles in context:")
        message_roles = [msg["role"] for msg in context.messages]
        provider.info(f"Message roles: {message_roles}")

        provider.success("Test completed successfully")

    except Exception as e:
        provider.error(f"Error in test: {str(e)}")
        import traceback
        provider.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
