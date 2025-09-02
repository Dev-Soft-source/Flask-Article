# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import argparse
import os
import sys
from typing import Optional
from rich.panel import Panel
from utils.rich_provider import provider
from rich.traceback import install

# Install rich traceback handler with showing locals
install(show_locals=True, width=None, word_wrap=False)

from config import Config, VOICE_TONES, ARTICLE_TYPES, ARTICLE_AUDIENCES, POINT_OF_VIEWS
from utils.prompts_config import Prompts
from utils.rate_limiter import initialize_rate_limiters, RateLimitConfig
from prompts_simplified import (
    TITLE_PROMPT,
    TITLE_CRAFT_PROMPT,
    OUTLINE_PROMPT,
    INTRODUCTION_PROMPT,
    PARAGRAPH_GENERATE_PROMPT,
    CONCLUSION_PROMPT,
    FAQ_PROMPT,
    SYSTEM_MESSAGE,
    GRAMMAR_CHECK_PROMPT,
    HUMANIZE_PROMPT,
    BLOCKNOTE_KEY_TAKEAWAYS_PROMPT,
    BLOCKNOTES_COMBINE_PROMPT,
    SUMMARIZE_PROMPT,
    SUMMARY_COMBINE_PROMPT,
    META_DESCRIPTION_PROMPT,
    WORDPRESS_META_DESCRIPTION_PROMPT,
    PAA_ANSWER_PROMPT,
)
from article_generator.generator import Generator
from article_generator.logger import logger

from article_generator.serpapi import validate_all_serpapi_keys


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Article Generation System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True, help="Path to input CSV file containing article data"
    )
    parser.add_argument(
        "--config", help="Path to custom configuration file (optional)", default=None
    )

    # API Keys
    parser.add_argument("--openai-key", help="OpenAI API key", default=None)
    parser.add_argument("--openai-model", help="OpenAI model to use", default=None)
    parser.add_argument("--serp-api-key", help="SerpAPI key", default=None)
    parser.add_argument("--unsplash-api-key", help="Unsplash API key", default=None)
    parser.add_argument("--youtube-api-key", help="YouTube API key", default=None)

    # OpenRouter Configuration
    parser.add_argument("--openrouter-key", help="OpenRouter API key", default=None)
    parser.add_argument(
        "--use-openrouter", help="Use OpenRouter instead of OpenAI", action="store_true"
    )
    parser.add_argument(
        "--openrouter-site-url",
        help="Your website URL for OpenRouter tracking",
        default=None,
    )
    parser.add_argument(
        "--openrouter-site-name",
        help="Your application name for OpenRouter tracking",
        default=None,
    )
    parser.add_argument(
        "--openrouter-model",
        help="OpenRouter model to use for content generation",
        default=None,
    )
    
    # Model selection for different features
    parser.add_argument(
        "--grammar-check-model", 
        help="Model to use for grammar checking",
        default=None,
    )
    parser.add_argument(
        "--use-separate-grammar-model", 
        help="Use a separate model for grammar checking",
        action="store_true",
    )
    parser.add_argument(
        "--humanization-model", 
        help="Model to use for text humanization",
        default=None,
    )
    parser.add_argument(
        "--use-separate-humanization-model", 
        help="Use a separate model for text humanization",
        action="store_true",
    )

    # WordPress Settings
    parser.add_argument("--wp-url", help="WordPress site URL", default=None)
    parser.add_argument("--wp-user", help="WordPress username", default=None)
    parser.add_argument(
        "--wp-app-pass", help="WordPress application password", default=None
    )

    # Article Configuration Settings
    parser.add_argument(
        "--voice-tone",
        help="Voice tone for the article",
        choices=VOICE_TONES.keys(),
        default="professional",
    )
    parser.add_argument(
        "--article-type",
        help="Type of article to generate",
        choices=ARTICLE_TYPES.keys(),
        default="Default",
    )
    parser.add_argument(
        "--article-audience",
        help="Target audience for the article",
        choices=ARTICLE_AUDIENCES.keys(),
        default="General",
    )
    parser.add_argument(
        "--point-of-view",
        help="Point of view for the article",
        choices=POINT_OF_VIEWS.keys(),
        default="Third Person",
    )

    return parser.parse_args()


def initialize_config() -> Optional[Config]:
    """Initialize and validate configuration."""
    try:
        logger.info("≫≫ [bold]Configuration Settings[/]")

        # Display available configuration options from dictionaries
        logger.info("[bold]Available Voice Tones:[/]")
        for tone, description in VOICE_TONES.items():
            logger.info(f"  • [cyan]{tone}:[/] {description}")

        logger.info("\n[bold]Available Article Types:[/]")
        for article_type, description in ARTICLE_TYPES.items():
            logger.info(f"  • [cyan]{article_type}:[/] {description}")

        logger.info("\n[bold]Available Article Audiences:[/]")
        for audience, description in ARTICLE_AUDIENCES.items():
            logger.info(f"  • [cyan]{audience}:[/] {description}")

        logger.info("\n[bold]Available Points of View:[/]")
        for pov, description in POINT_OF_VIEWS.items():
            logger.info(f"  • [cyan]{pov}:[/] {description}")

        logger.info("\n[bold]Initial Configuration Settings:[/]")

        best_key, all_keys = validate_all_serpapi_keys()

        # Initialize configuration with environment variables
        config = Config(
            # API Keys and Credentials are loaded from environment variables
            openai_model="gpt-4o-mini-2024-07-18",
            serp_api_key=best_key,
            
            # OpenRouter Integration
            use_openrouter=True,
            openrouter_model="deepseek/deepseek-chat-v3-0324:free",
            rag_openrouter_model="google/gemini-2.0-flash-exp:free",
            
            # Grammar and Humanization Models
            enable_separate_grammar_model=True,
            grammar_check_model="google/gemini-2.0-flash-exp:free",
            enable_separate_humanization_model=True,
            humanization_model="google/gemini-2.0-flash-exp:free",

            # RAG Settings
            enable_rag=False,
            rag_chunk_size=500,
            rag_num_chunks=3,
            rag_cache_dir="cache/rag_cache",
            rag_embedding_model="all-MiniLM-L6-v2",
            rag_embedding_dimension=384,
            enable_rag_search_engine=False,
            rag_article_retriever_engine="Duckduckgo",
            
            # Article Settings
            articlelanguage="English",
            articleaudience="General",  # From ARTICLE_AUDIENCES dictionary: General - Accessible to most readers with minimal specialized knowledge
            articletype="Default",  # From ARTICLE_TYPES dictionary: Default - Standard informational article
            
            # Voice and Tone Settings
            voicetone="friendly",  # From VOICE_TONES dictionary: friendly - Warm, approachable style that connects personally with the reader
            pointofview="Third Person",  # From POINT_OF_VIEWS dictionary: Third Person - Objective perspective using he, she, they, or it
            sizesections=2,  # Number of sections
            sizeheadings=3,  # Number of subsections
            paragraphs_per_section=2,  # Number of paragraphs per section
            
            # Feature Toggles
            add_summary_into_article=False,
            add_faq_into_article=False,
            add_image_into_article=False,
            enable_image_generation=False,
            add_youtube_video=False,
            add_external_links_into_article=False,
            add_paa_paragraphs_into_article=True,
            add_blocknote_into_article=True,

            enable_grammar_check=False,
            enable_text_humanization=False,
            
            enable_progress_display=True,
            enable_token_tracking=True,
            enable_wordpress_upload=True,
            enable_markdown_save=True,
            enable_meta_description=True,
            
            # Seed Control Settings
            # enable_seed_control=True,  # Enable seed control
            # title_seed=123,
            # outline_seed=456,
            # introduction_seed=456,
            # paragraph_seed=789,
            # conclusion_seed=101,
            # key_takeaways_seed=112,
            # faq_seed=123,
            # meta_description_seed=134,
            
            # Meta Description Settings
            meta_description_max_length=155,
            meta_description_min_length=120,
            meta_description_temperature=0.7,
            meta_description_top_p=1.0,
            meta_description_frequency_penalty=0.0,
            meta_description_presence_penalty=0.0,
            
            # Image Settings
            max_number_of_images=24,
            orientation="landscape",
            order_by="relevant",
            
            # Generation Control Settings
            content_generation_temperature=1.0,
            content_generation_top_p=1.0,
            content_generation_frequency_penalty=0.0,
            content_generation_presence_penalty=0.0,
            humanization_temperature=0.7,
            humanization_top_p=1.0,
            humanization_frequency_penalty=0.0,
            humanization_presence_penalty=0.0,
            grammar_check_temperature=0.3,
            grammar_check_top_p=1.0,
            grammar_check_frequency_penalty=0.0,
            grammar_check_presence_penalty=0.0,
            block_notes_temperature=0.7,
            block_notes_top_p=1.0,
            block_notes_frequency_penalty=0.0,
            block_notes_presence_penalty=0.0,
            faq_generation_temperature=0.7,
            faq_generation_top_p=1.0,
            faq_generation_frequency_penalty=0.0,
            faq_generation_presence_penalty=0.0,

            enable_paragraph_headings=True,  # Whether to generate headings for each paragraph

            # Error Handling Settings
            initial_delay=1,
            exponential_base=2,
            jitter=True,
            max_retries=10,
            
            # Token Settings
            max_context_window_tokens=128000,
            token_padding=1000,
            warn_token_threshold=0.9,
            
            # Output Settings
            markdown_output_dir="generated_articles",
            
            # Rate Limiting Settings
            enable_rate_limiting=True,
            openai_rpm=60,
            openai_rpd=10000,
            serpapi_rpm=5,
            serpapi_rpd=100,
            unsplash_rpm=50,
            unsplash_rpd=5000,
            youtube_rpm=100,
            youtube_rpd=10000,
            rate_limit_cooldown=60,
            
            # positioning
            youtube_position = "random", #random / first_heading / end
            keytakeaways_position = "random", #random / before_conclusion / middle
            image_position = "random", #random / under_first heading /middle / end
            paa_image_position = "middle",  # first_heading / middle
            
            
            image_source = "stock",
            stock_primary_source = "openverse",
            secondary_source_image= True,
            image_api=True,
            huggingface_model="stabilityai/stable-diffusion-xl-base-1.0",
            
            # Image captioning settings 
            image_caption_instance="openai/clip-vit-base-patch32",
            
            # Image sources api keys
            unsplash_api_key=os.getenv('UNSPLASH_API_KEY',''),
            pexels_api_key=os.getenv('PEXELS_API_KEY',''),
            pixabay_api_key=os.getenv('PIXABAY_API_KEY',''),
            giphy_api_key=os.getenv('GIPHY_API_KEY',''),
            huggingface_api_key=os.getenv('HUGGINGFACE_API_KEY',''),
            
             # Openverse settings (as of 2024)
            openverse_rpm=60,  # Openverse unofficially allows ~60 requests per minute
            openverse_rpd=5000,# Conservative daily limit; official cap isn't enforced publicly

            # Pexels settings (as of 2024)
            pexels_rpm=200,     # Official: 200 requests per minute
            pexels_rpd=20000, # Official: 20,000 requests per month for free tier → ~666/day

            # Pixabay settings (as of 2024)
            pixabay_rpm=60,    # Unofficial safe limit
            pixabay_rpd=5000, # Default daily quota for free-tier API keys

            # DuckDuckGo settings (as of 2024)
            duckduckgo_rpm=30,      
            duckduckgo_rpd=10000,
            
            # Huggingface settings (as of 2024)
            huggingface_rpm=30 ,     
            huggingface_rpd=10000,
            
            # Context Save Settings
            enable_context_save=True,  # Enable saving ArticleContext to a file
            context_save_dir="article_contexts",  # Directory to save context files
            
            # Summary and Keynotes Model Settings
            enable_separate_summary_model=True,  # Enable using a separate model for summary and keynotes
            summary_keynotes_model="mistralai/mistral-small-3.2-24b-instruct:free",  # Default model with large context window
            summary_max_tokens=800,  # Maximum tokens for summary generation
            keynotes_max_tokens=300,  # Maximum tokens for keynotes generation
            summary_chunk_size=10e3,  # Chunk size for summary generation when chunking is needed
            keynotes_chunk_size=10e3,  # Chunk size for keynotes generation when chunking is needed
            summary_combination_temperature=0.4, # Temperature for combining summary chunks
            summary_combination_paragraphs=2, # Number of paragraphs to combine when chunking
            keynotes_combination_temperature = 0.5,  # Temperature for combining keynotes chunks
            keynotes_combination_paragraphs = 1,  # Number of paragraphs to generate when combining keynotes chunks
            
            # WordPress Settings
            WP_WEBSITE_NAME=os.getenv('WP_WEBSITE_NAME', ''),
            WP_USERNAME=os.getenv('WP_USERNAME', ''),
            wp_app_pass=os.getenv('WP_APP_PASS', ''),
            wp_categories="1",  # Default category ID
            wp_author="1",  # Default author ID
            wp_custom_author="",  # Optional custom author ID
            wp_post_status="draft",  # Default post status (draft or publish)
        )

        # Initialize rate limiters if enabled
        if config.enable_rate_limiting:
            logger.info("Initializing rate limiters...")
            initialize_rate_limiters(
                openai_config=RateLimitConfig(
                    rpm=config.openai_rpm,
                    rpd=config.openai_rpd,
                    cooldown_period=config.rate_limit_cooldown,
                    enabled=config.enable_rate_limiting,
                ),
                serpapi_config=RateLimitConfig(
                    rpm=config.serpapi_rpm,
                    rpd=config.serpapi_rpd,
                    cooldown_period=config.rate_limit_cooldown,
                    enabled=config.enable_rate_limiting,
                ),
                duckduckgo_config=RateLimitConfig(
                    rpm=config.duckduckgo_rpm,
                    rpd=config.duckduckgo_rpd,
                    cooldown_period=config.rate_limit_cooldown,
                    enabled=config.enable_rate_limiting
                ),
                unsplash_config=RateLimitConfig(
                    rpm=config.unsplash_rpm,
                    rpd=config.unsplash_rpd,
                    cooldown_period=config.rate_limit_cooldown,
                    enabled=config.enable_rate_limiting,
                ),
                youtube_config=RateLimitConfig(
                    rpm=config.youtube_rpm,
                    rpd=config.youtube_rpd,
                    cooldown_period=config.rate_limit_cooldown,
                    enabled=config.enable_rate_limiting,
                ),
                openverse_config=RateLimitConfig(
                    rpm=config.openverse_rpm,
                    rpd=config.openverse_rpd,
                    cooldown_period=config.rate_limit_cooldown,
                    enabled=config.enable_rate_limiting
                ),
                pexels_config=RateLimitConfig(
                    rpm=config.pexels_rpm,
                    rpd=config.pexels_rpd,
                    cooldown_period=config.rate_limit_cooldown,
                    enabled=config.enable_rate_limiting
                ),
                pixabay_config=RateLimitConfig(
                    rpm=config.pixabay_rpm,
                    rpd=config.pixabay_rpd,
                    cooldown_period=config.rate_limit_cooldown,
                    enabled=config.enable_rate_limiting
                ),
                huggingface_config=RateLimitConfig(
                    rpm=config.huggingface_rpm,
                    rpd=config.huggingface_rpd,
                    cooldown_period=config.rate_limit_cooldown,
                    enabled=config.enable_rate_limiting
                )
            )
            logger.success("✓ Rate limiters initialized successfully")

        # Display selected configuration values
        logger.success(f"✓ Voice Tone: [green]{config.voicetone}[/] - {config.available_voice_tones[config.voicetone]}")
        logger.success(f"✓ Article Type: [green]{config.articletype}[/] - {config.available_article_types[config.articletype]}")
        logger.success(f"✓ Audience: [green]{config.articleaudience}[/] - {config.available_article_audiences[config.articleaudience]}")
        logger.success(f"✓ POV: [green]{config.pointofview}[/] - {config.available_point_of_views[config.pointofview]}")
        
        # Display WordPress author information if custom author is being used
        if config.wp_custom_author:
            logger.info(f"[bold yellow]Using custom WordPress author ID: {config.wp_custom_author}[/] (overriding default: {config.wp_author})")

        logger.success("\n✓ Configuration loaded successfully")
        return config

    except Exception as e:
        logger.error(f"Failed to initialize configuration: {str(e)}", show_traceback=True)
        return None


def initialize_prompts() -> Prompts:
    """Initialize prompts configuration."""
    return Prompts(
        title=TITLE_PROMPT,
        title_craft=TITLE_CRAFT_PROMPT,
        outline=OUTLINE_PROMPT,
        introduction=INTRODUCTION_PROMPT,
        paragraph=PARAGRAPH_GENERATE_PROMPT,
        paragraph_with_heading=PARAGRAPH_GENERATE_PROMPT,
        conclusion=CONCLUSION_PROMPT,
        faq=FAQ_PROMPT,
        system_message=SYSTEM_MESSAGE,
        grammar=GRAMMAR_CHECK_PROMPT,
        humanize=HUMANIZE_PROMPT,
        blocknote=BLOCKNOTE_KEY_TAKEAWAYS_PROMPT,
        summarize=SUMMARIZE_PROMPT,
        summary_combine=SUMMARY_COMBINE_PROMPT,
        meta_description=META_DESCRIPTION_PROMPT,
        wordpress_excerpt=WORDPRESS_META_DESCRIPTION_PROMPT,
        paa_answer=PAA_ANSWER_PROMPT,
        blocknotes_combine=BLOCKNOTES_COMBINE_PROMPT,
    )


def process_articles(generator: Generator, csv_path: str) -> bool:
    """
    Process all articles from the CSV file.

    Args:
        generator (Generator): Initialized article generator
        csv_path (str): Path to the CSV file

    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Process CSV file
        if not generator.process_csv(csv_path):
            return False

        # Get total number of articles
        total_articles = generator.csv_processor.get_total_articles()
        successful_generations = 0

        # Create progress bar using provider
        with provider.progress_bar() as progress:
            task = progress.add_task("Generating articles...", total=total_articles)

            # Process each article
            for index in range(1, total_articles + 1):
                article_data = generator.csv_processor.get_article_data(index)
                if not article_data:
                    logger.warning(f"No data found for article {index}, skipping...")
                    progress.update(task, advance=1)
                    continue

                logger.info(f"[bold]Processing article {index}/{total_articles}[/]")

                try:
                    # Generate article
                    article = generator.generate_article(article_data)
                    if article:
                        successful_generations += 1
                        logger.success(f"✓ Article {index} generated successfully")
                    else:
                        logger.error(f"Failed to generate article {index}")

                    progress.update(task, advance=1)
                except Exception as article_error:
                    logger.error(f"Error generating article {index}: {str(article_error)}", show_traceback=True)
                    progress.update(task, advance=1)

            # Display final summary
            logger.info(f"\n[bold]Generation Summary[/]")
            logger.info(f"• Successful Generations: [green]{successful_generations}[/]")
            logger.info(f"• Failed Generations: [red]{total_articles - successful_generations}[/]")
            logger.info(f"• Total Articles: [blue]{total_articles}[/]")

            return successful_generations > 0

    except Exception as e:
        logger.error(f"Error processing articles: {str(e)}", show_traceback=True)
        return False


def main() -> int:
    """Main entry point for the article generation system."""
    try:
        provider.console.print(
            Panel.fit(
                "[bold blue]AI Article Generation System[/]",
                subtitle="Powered by OpenAI GPT",
                border_style="blue",
            )
        )

        # Parse command line arguments
        args = parse_arguments()

        # Initialize configuration
        config = initialize_config()
        if not config:
            logger.error("✗ Configuration initialization failed. Exiting.")
            return 1

        # Override configuration with command line arguments if provided
        if args.openai_key:
            config.openai_key = args.openai_key
            logger.debug("Using OpenAI API key from command line")

        if args.openai_model:
            config.openai_model = args.openai_model
            logger.debug(f"Using OpenAI model from command line: {args.openai_model}")

        if args.serp_api_key:
            config.serp_api_key = args.serp_api_key
            logger.debug("Using SerpAPI key from command line")

        # if args.image_api_key:
        #     config.image_api_key = args.image_api_key
        #     logger.debug("Using Unsplash API key from command line")

        if args.youtube_api_key:
            config.youtube_api_key = args.youtube_api_key
            logger.debug("Using YouTube API key from command line")

        if args.wp_url:
            config.WP_WEBSITE_NAME = args.wp_url
            logger.debug(f"Using WordPress URL from command line: {args.wp_url}")

        if args.wp_user:
            config.WP_USERNAME = args.wp_user
            logger.debug(f"Using WordPress username from command line: {args.wp_user}")

        if args.wp_app_pass:
            config.wp_app_pass = args.wp_app_pass
            logger.debug("Using WordPress application password from command line")

        # OpenRouter settings
        if args.openrouter_key:
            config.openrouter_api_key = args.openrouter_key
        if args.use_openrouter:
            config.use_openrouter = True
        if args.openrouter_site_url:
            config.openrouter_site_url = args.openrouter_site_url
        if args.openrouter_site_name:
            config.openrouter_site_name = args.openrouter_site_name
        if args.openrouter_model:
            config.openrouter_model = args.openrouter_model
            
        # Model selection settings
        if args.grammar_check_model:
            config.grammar_check_model = args.grammar_check_model
        if args.use_separate_grammar_model:
            config.enable_separate_grammar_model = True
        if args.humanization_model:
            config.humanization_model = args.humanization_model
        if args.use_separate_humanization_model:
            config.enable_separate_humanization_model = True
        if args.openrouter_site_name:
            config.openrouter_site_name = args.openrouter_site_name

        # Override article configuration settings with command line arguments
        if args.voice_tone:
            config.voicetone = args.voice_tone
            logger.debug(f"Using voice tone from command line: {args.voice_tone} - {VOICE_TONES[args.voice_tone]}")

        if args.article_type:
            config.articletype = args.article_type
            logger.debug(f"Using article type from command line: {args.article_type} - {ARTICLE_TYPES[args.article_type]}")

        if args.article_audience:
            config.articleaudience = args.article_audience
            logger.debug(f"Using article audience from command line: {args.article_audience} - {ARTICLE_AUDIENCES[args.article_audience]}")

        if args.point_of_view:
            config.pointofview = args.point_of_view
            logger.debug(f"Using point of view from command line: {args.point_of_view} - {POINT_OF_VIEWS[args.point_of_view]}")

        # Initialize prompts
        prompts = initialize_prompts()

        # Create article generator
        generator = Generator(config, prompts)

        # Validate APIs
        if not generator.validate_apis():
            logger.warning("⚠ Some API validations failed. Continuing with limited functionality.")

        # Process articles
        if not process_articles(generator, args.input):
            logger.error("✗ Article generation failed. Check logs for details.")
            return 1

        logger.success("✓ Article generation completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠ Process interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unhandled exception in main: {str(e)}", show_traceback=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
