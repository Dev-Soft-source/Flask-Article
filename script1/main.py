# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.traceback import install
import sys
import os
from dotenv import load_dotenv
from utils.error_utils import ErrorHandler

# Install rich traceback handler with showing locals
install(show_locals=True, width=None, word_wrap=False)

# Initialize error handler
error_handler = ErrorHandler()

# Load environment variables from .env file
load_dotenv()

from utils.config import (
    Config,
    VOICE_TONES,
    ARTICLE_TYPES,
    ARTICLE_AUDIENCES,
    POINT_OF_VIEWS
)
from article_generator.generator import Generator, Prompts
from prompts_simplified import (
    TITLE_PROMPT,
    TITLE_CRAFT_PROMPT,
    FIXED_OUTLINE_PROMPT,
    VARIABLE_PARAGRAPHS_OUTLINE_PROMPT,
    TWO_LEVEL_OUTLINE_PROMPT,
    VARIABLE_TWO_LEVEL_OUTLINE_PROMPT,
    INTRODUCTION_PROMPT,
    PARAGRAPH_PROMPT,
    CONCLUSION_PROMPT,
    HUMANIZE_PROMPT,
    SUMMARIZE_PROMPT,
    SUMMARY_COMBINE_PROMPT,
    FAQ_PROMPT,
    BLOCKNOTE_KEY_TAKEAWAYS_PROMPT,
    META_DESCRIPTION_PROMPT,
    WORDPRESS_META_DESCRIPTION_PROMPT,
    GRAMMAR_CHECK_PROMPT,
    PAA_ANSWER_PROMPT,
)

from article_generator.serpapi import validate_all_serpapi_keys

# Initialize Rich console and global variables
console = Console()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Article Generator - AI-powered content creation system",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to input CSV file containing keywords (format: keyword,image_keyword)"
    )
    
    # API Keys
    parser.add_argument("--openai-key", help="OpenAI API key", default=None)
    parser.add_argument("--openai-model", help="OpenAI model to use", default=None)
    parser.add_argument("--serp-api-key", help="SerpAPI key", default=None)
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
        default=None,
    )
    parser.add_argument(
        "--article-type",
        help="Type of article to generate",
        choices=ARTICLE_TYPES.keys(),
        default=None,
    )
    parser.add_argument(
        "--article-audience",
        help="Target audience for the article",
        choices=ARTICLE_AUDIENCES.keys(),
        default=None,
    )
    parser.add_argument(
        "--point-of-view",
        help="Point of view for the article",
        choices=POINT_OF_VIEWS.keys(),
        default=None,
    )
    
    return parser.parse_args()


def main():
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        console.print(
            Panel("[bold blue]Article Generator[/]", subtitle="Powered by OpenAI GPT")
        )

        # Display available configuration options
        console.print("[bold cyan]Available Voice Tones:[/]")
        for tone, description in VOICE_TONES.items():
            console.print(f"  - [yellow]{tone}[/]: {description}")

        console.print("\n[bold cyan]Available Article Types:[/]")
        for article_type, description in ARTICLE_TYPES.items():
            console.print(f"  - [yellow]{article_type}[/]: {description}")

        console.print("\n[bold cyan]Available Article Audiences:[/]")
        for audience, description in ARTICLE_AUDIENCES.items():
            console.print(f"  - [yellow]{audience}[/]: {description}")

        console.print("\n[bold cyan]Available Points of View:[/]")
        for pov, description in POINT_OF_VIEWS.items():
            console.print(f"  - [yellow]{pov}[/]: {description}")

        console.print("\n[bold green]Using the following configuration settings:[/]")
        best_key, all_keys = validate_all_serpapi_keys()

        # Initialize config with environment variables
        config = Config(
            # API Keys and Credentials are loaded from environment variables
            openai_model="gpt-4o-mini-2024-07-18",
            openrouter_model="deepseek/deepseek-chat-v3-0324:free",
            rag_openrouter_model="google/gemini-2.0-flash-exp:free",

            # Grammar and Humanization Models
            enable_separate_grammar_model=False,
            grammar_check_model="google/gemini-2.0-flash-exp:free",
            enable_separate_humanization_model=True,
            humanization_model="google/gemini-2.0-flash-exp:free",

            # Use OpenRouter instead of OpenAI
            use_openrouter=True,

            # Override serp_api_key with the best key from validation
            serp_api_key=best_key,

            # Voice Tone Settings
            voicetone="friendly",  # From VOICE_TONES dictionary: friendly - Warm, approachable style that connects personally with the reader

            # Article Type Settings
            articletype="Default",  # From ARTICLE_TYPES dictionary: Default - Standard informational article

            # Article Audience Settings
            articleaudience="General",  # From ARTICLE_AUDIENCES dictionary: General - Accessible to most readers with minimal specialized knowledge

            # Language and Point of View Settings
            pointofview="Third Person",  # From POINT_OF_VIEWS dictionary: Third Person - Objective perspective using 'he,' 'she,' 'they,' or 'it'
            articlelanguage="English",

            # Article Structure Settings
            sizesections=2,  # Number of sections in the article
            sizeheadings=0, # Number of headings in each section (subheadings, basically)
            enable_variable_paragraphs=False,

            # Article Size Settings
            section_token_limit=2000,
            paragraphs_per_section=2,
            min_paragraph_tokens=300,
            max_paragraph_tokens=500,

            # Bold and Italics, etc into Articles Settings
            add_bold_into_article = True,
            add_lists_into_articles = True,
            add_tables_into_articles = True,
            enable_variable_subheadings = True,

            
            # Image Settings
            image_source = "stock",
            stock_primary_source = "pixabay",
            secondary_source_image= True,
            image_api=True,
            huggingface_model="stabilityai/stable-diffusion-xl-base-1.0",
            
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
            
            # Openverse settings (as of 2024)
            openverse_rpm=60,
            openverse_rpd=5000,
            
            # Pexels settings (as of 2024)
            pexels_rpm=200,
            pexels_rpd=20000,
            
            # Pixabay settings (as of 2024)
            pixabay_rpm=60,
            pixabay_rpd=5000,
            
            # DuckDuckGo settings (as of 2024)
            duckduckgo_rpm=30,
            duckduckgo_rpd=10000,
            
            # Huggingface settings (as of 2024)
            huggingface_rpm=30,
            huggingface_rpd=10000,
            
            # Image sources api keys
            unsplash_api_key=os.getenv('UNSPLASH_API_KEY',''),
            pexels_api_key=os.getenv('PEXELS_API_KEY',''),
            pixabay_api_key=os.getenv('PIXABAY_API_KEY',''),
            giphy_api_key=os.getenv('GIPHY_API_KEY',''),
            huggingface_api_key=os.getenv('HUGGINGFACE_API_KEY',''),
            
            # Image captioning instance
            image_caption_instance="openai/clip-vit-base-patch32",
            
            # positioning
            youtube_position = "random", #random / first_heading / end
            keytakeaways_position = "random", #random / before_conclusion / middle
            image_position = "end", #random / under_first_heading /middle / end
            paa_image_position = "random", # random / first_heading / end


            # Article Generation Settings
            # Image
            add_image_into_article=False,
            enable_image_generation=False,
            # Other settings
            add_blocknote_into_article=True,
            add_summary_into_article=True,
            add_external_links_into_article=True,
            add_PAA_paragraphs_into_article=True,
            add_faq_into_article=False,  # Whether to add FAQ section into the article   
            
            enable_paragraph_headings=True,  # Whether to generate headings for each paragraph

            # Meta Description Settings
            enable_meta_description=True,
            meta_description_max_length=155,
            meta_description_min_length=120,

            # Feature Toggles
            enable_wordpress_upload=True,
            enable_text_humanization=False,
            enable_grammar_check=False,
            enable_youtube_videos=False,

            # WordPress Settings
            website_name=os.getenv('WP_WEBSITE_NAME', ''),
            Username=os.getenv('WP_USERNAME', ''),
            App_pass=os.getenv('WP_APP_PASS', ''),
            categories="1",  # Default category ID
            author="1",  # Default author ID
            custom_author="",  # Optional custom author ID
            status="draft",  # Default post status (draft or publish)

            # Temperature Settings
            meta_description_temperature=0.7,
            content_generation_temperature=1.0,
            humanization_temperature=0.7,
            grammar_check_temperature=0.3,
            block_notes_temperature=0.7,
            faq_generation_temperature=1.0,

            # Top P Settings
            meta_description_top_p=1.0,
            content_generation_top_p=1.0,
            humanization_top_p=1.0,
            grammar_check_top_p=1.0,
            block_notes_top_p=1.0,
            faq_generation_top_p=1.0,

            # Frequency Penalty Settings
            meta_description_frequency_penalty=0.0,
            content_generation_frequency_penalty=0.0,
            humanization_frequency_penalty=0.0,
            grammar_check_frequency_penalty=0.0,
            block_notes_frequency_penalty=0.0,
            faq_generation_frequency_penalty=0.0,
            
            # Seed Control Settings
            enable_seed_control=False,
            enable_rag_search_engine=False,
            # title_seed=123,
            # outline_seed=456,
            # introduction_seed=456,
            # paragraph_seed=789,
            # conclusion_seed=101,
            # key_takeaways_seed=112,
            # faq_seed=123,
            # meta_description_seed=134,

            # Context Save Settings
            enable_context_save=True,  # Enable saving ArticleContext to a file
            context_save_dir="article_contexts",  # Directory to save context files

            # RAG Configuration
            enable_rag=False,
            rag_chunk_size=500,
            rag_num_chunks=3,
            rag_cache_dir="cache/rag_cache",
            rag_embedding_model="all-MiniLM-L6-v2",
            rag_embedding_dimension=384,
            rag_article_retriever_engine="Duckduckgo",

            enable_separate_summary_model=True, # Enable using a separate model for summary and keynotes
            summary_keynotes_model="mistralai/mistral-small-3.2-24b-instruct:free", # Default model with large context window
            summary_max_tokens=800, # Maximum tokens for summary generation
            keynotes_max_tokens=300, # Maximum tokens for keynotes generation
            summary_chunk_size=10_000, # Chunk size for summary generation when chunking is needed
            keynotes_chunk_size=10_000, # Chunk size for keynotes generation when chunking is needed
            summary_combination_temperature=0.4, # Temperature for combining summary chunks
            summary_combination_paragraphs=2, # Number of paragraphs to combine when chunking
            keynotes_combination_temperature = 0.5,  # Temperature for combining keynotes chunks
            keynotes_combination_paragraphs = 1,  # Number of paragraphs to generate when combining keynotes chunks
    
            # PAA Settings
            paa_max_questions = 5,  # Maximum number of PAA questions to display
            paa_min_questions = 3,  # Minimum number of PAA questions when using random range
            paa_use_random_range = False,  # Whether to use a random range for PAA questions

          
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

          
            # Enhanced image settings
            max_number_of_images=24,
            orientation="landscape",
            order_by="relevant",
          

            # Feature toggles for enhanced control
            enable_progress_display=True,
            enable_token_tracking=True,
            enable_markdown_save=True,
            
            # Presence and frequency penalties
            meta_description_presence_penalty=0.0,
            content_generation_presence_penalty=0.0,
            humanization_presence_penalty=0.0,
            grammar_check_presence_penalty=0.0,
            block_notes_presence_penalty=0.0,
            faq_generation_presence_penalty=0.0,
        )

        # Display selected configuration values
        console.print(f"[cyan]Selected Voice Tone:[/] [yellow]{config.voicetone}[/] - {config.available_voice_tones[config.voicetone]}")
        console.print(f"[cyan]Selected Article Type:[/] [yellow]{config.articletype}[/] - {config.available_article_types[config.articletype]}")
        console.print(f"[cyan]Selected Article Audience:[/] [yellow]{config.articleaudience}[/] - {config.available_article_audiences[config.articleaudience]}")
        console.print(f"[cyan]Selected Point of View:[/] [yellow]{config.pointofview}[/] - {config.available_point_of_views[config.pointofview]}")

        # Override configuration with command line arguments if provided
        if args.openai_key:
            config.openai_key = args.openai_key
            console.print("[dim]Using OpenAI API key from command line[/]")

        if args.openai_model:
            config.openai_model = args.openai_model
            console.print(f"[dim]Using OpenAI model from command line: {args.openai_model}[/]")

        if args.serp_api_key:
            config.serp_api_key = args.serp_api_key
            console.print("[dim]Using SerpAPI key from command line[/]")

        if args.wp_url:
            config.website_name = args.wp_url
            console.print(f"[dim]Using WordPress URL from command line: {args.wp_url}[/]")

        if args.wp_user:
            config.Username = args.wp_user
            console.print(f"[dim]Using WordPress username from command line: {args.wp_user}[/]")

        if args.wp_app_pass:
            config.App_pass = args.wp_app_pass
            console.print("[dim]Using WordPress application password from command line[/]")

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

        # Override article configuration settings with command line arguments
        if args.voice_tone:
            config.voicetone = args.voice_tone
            console.print(f"[dim]Using voice tone from command line: {args.voice_tone}[/]")

        if args.article_type:
            config.articletype = args.article_type
            console.print(f"[dim]Using article type from command line: {args.article_type}[/]")

        if args.article_audience:
            config.articleaudience = args.article_audience
            console.print(f"[dim]Using article audience from command line: {args.article_audience}[/]")

        if args.point_of_view:
            config.pointofview = args.point_of_view
            console.print(f"[dim]Using point of view from command line: {args.point_of_view}[/]")

        # Initialize prompts
        prompts = Prompts(
            title=TITLE_PROMPT,
            title_craft=TITLE_CRAFT_PROMPT,
            fixed_outline=FIXED_OUTLINE_PROMPT,
            variable_paragraphs_outline=VARIABLE_PARAGRAPHS_OUTLINE_PROMPT,
            two_level_outline=TWO_LEVEL_OUTLINE_PROMPT,
            variable_two_level_outline=VARIABLE_TWO_LEVEL_OUTLINE_PROMPT,
            introduction=INTRODUCTION_PROMPT,
            paragraph_generate=PARAGRAPH_PROMPT,
            conclusion=CONCLUSION_PROMPT,
            humanize=HUMANIZE_PROMPT,
            summarize=SUMMARIZE_PROMPT,
            summary_combine=SUMMARY_COMBINE_PROMPT,
            faqs=FAQ_PROMPT,
            blocknote=BLOCKNOTE_KEY_TAKEAWAYS_PROMPT,
            grammar=GRAMMAR_CHECK_PROMPT,
            meta_description=META_DESCRIPTION_PROMPT,
            wordpress_excerpt=WORDPRESS_META_DESCRIPTION_PROMPT,
            paa_answer=PAA_ANSWER_PROMPT,
        )

        # Initialize generator
        generator = Generator(config, prompts)

        # Set up environment
        if not generator.setup_environment():
            error_handler.handle_error(Exception("Failed to set up environment"), severity="error")
            return

        # Get keywords from file
        keywords_file = args.input
        try:
            # Use the improved read_keywords_file function that uses UnifiedCSVProcessor
            from utils.file_utils import read_keywords_file
            keyword_pairs = read_keywords_file(keywords_file)

            if not keyword_pairs:
                error_handler.handle_error(Exception("No valid keywords found in the file"), severity="error")
                error_handler.handle_error(Exception("Please ensure your CSV file has the correct format"), severity="warning")
                console.print("keyword,image_keyword")
                console.print("how to grow tomatoes,tomato plant")
                console.print("how to make pasta,pasta cooking")
                return

            console.print(f"\n[bold cyan]Found {len(keyword_pairs)} keywords to process.[/]")
            console.print("[bold green]Will now create articles with the main keywords.[/]")

            # Process each keyword pair
            for idx, (keyword, image_keyword) in enumerate(keyword_pairs, start=1):
                console.print(f"\n[bold]Processing article {idx}/{len(keyword_pairs)}: {keyword}[/]")

                try:
                    # Check if a specific author ID was provided as an environment variable
                    custom_author_id = os.getenv('WP_CUSTOM_AUTHOR')
                    if custom_author_id:
                        console.print(f"[cyan]Using custom WordPress author ID: {custom_author_id}[/]")
                        
                    # Generate article with optional custom author ID
                    article = generator.generate_article(keyword, image_keyword, custom_author_id)
                    # print(article)
                    console.print(
                        f"[green]✓[/] Successfully generated article for: {keyword}"
                    )
                except Exception as e:
                    console.print(
                        f"[red]✗[/] Failed to generate article for {keyword}: {str(e)}"
                    )
        except Exception as e:
            error_handler.handle_error(e, severity="error")
            error_handler.handle_error(Exception("Please ensure your CSV file has the correct format"), severity="warning")
            console.print("keyword,image_keyword")
            console.print("how to grow tomatoes,tomato plant")
            console.print("how to make pasta,pasta cooking")
            return

    except Exception as e:
        error_handler.handle_error(e, severity="error")
        raise


if __name__ == "__main__":
    main()
