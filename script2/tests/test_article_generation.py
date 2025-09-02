# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import os
import sys
from config import Config
from utils.prompts_config import Prompts
from article_generator.content_generator import ContentGenerator
from article_generator.article_context import ArticleContext
from utils.rich_provider import provider

def main():
    """Test article generation with detailed context logging."""
    try:
        provider.info("Testing article generation with detailed context logging...")

        # Create a test config with context saving enabled
        config = Config(
            # API settings
            openai_model="gpt-4o-mini-2024-07-18",
            use_openrouter=True,
            openrouter_model="meta-llama/llama-3.3-70b-instruct:free",

            # Article settings
            articlelanguage="English",
            articleaudience="General",
            articletype="Default",
            sizesections=1,  # Small article for testing
            sizeheadings=1,
            paragraphs_per_section=1,

            # Token settings
            max_context_window_tokens=128000,
            token_padding=1000,
            warn_token_threshold=0.9,
        )

        # Create prompts
        prompts = Prompts(
            system_message="You are an expert content writer creating cohesive, engaging articles.",
            title="Generate a title for an article about {keyword}.",
            outline="Create an outline for an article about {keyword}.",
            introduction="Write an introduction for an article about {keyword}.",
            paragraph="Write a paragraph about {subtitle} related to {keyword}.",
            conclusion="Write a conclusion for an article about {keyword}.",
            faq="Generate frequently asked questions about {keyword}.",
            meta_description="Write a meta description for an article about {keyword}.",
            wordpress_excerpt="Write a WordPress excerpt for an article about {keyword}.",
            grammar="Fix any grammar issues in the following text: {text}",
            humanize="Make this text more engaging and natural: {text}",
            blocknote="Create notes for each section of this article about {keyword}.",
            summarize="Summarize this article about {keyword}.",
            paa_answer="Answer the question: {question} related to {keyword}.",
            summary_combine="Combine these summary chunks into a cohesive summary.",
            blocknotes_combine="Combine these block note chunks into cohesive notes."
        )

        # Create article context
        context = ArticleContext(config, prompts)

        # Create content generator
        generator = ContentGenerator(config, prompts)
        generator.context = context  # Set the context after initialization

        # Generate a simple article
        provider.info("Testing content generation with detailed context logging...")

        # Use a simple keyword for testing
        keyword = "artificial intelligence basics"

        # Generate title
        provider.info("Generating title...")
        title = generator.generate_title(keyword)
        provider.success(f"Generated title: {title}")

        # Generate outline
        provider.info("Generating outline...")
        outline = generator.generate_outline(keyword)
        provider.success(f"Generated outline")

        # Generate introduction
        provider.info("Generating introduction...")
        intro = generator.generate_introduction(keyword, title)
        provider.success(f"Generated introduction")

        # Generate a paragraph
        provider.info("Generating paragraph...")
        paragraph = generator.generate_paragraph(
            keyword, 
            "Understanding AI Basics",
            current_paragraph=1,
            paragraphs_per_section=2,
            section_number=1,
            total_sections=3,
            section_points=["Basic concepts of AI", "Historical development of AI"]
        )
        provider.success(f"Generated paragraph")

        # Generate conclusion
        provider.info("Generating conclusion...")
        conclusion = generator.generate_conclusion(keyword, title)
        provider.success(f"Generated conclusion")

        # Save context to file
        filepath = context.save_to_file("test_context.md")

        provider.success("Content generation completed!")
        provider.info(f"Context saved to: {filepath}")
        provider.info(f"Number of messages in context: {len(context.messages)}")

        # Print message roles to verify
        message_roles = [msg["role"] for msg in context.messages]
        provider.info(f"Message roles: {message_roles}")

    except Exception as e:
        provider.error(f"Error in test: {str(e)}")
        import traceback
        provider.error(f"Traceback:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()
