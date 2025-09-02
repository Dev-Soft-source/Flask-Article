# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import base64
from typing import Dict, List, Optional, Tuple
import os
import time
import openai
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TaskID

from utils.prompts_config import Prompts
from utils.config import Config
from article_generator.content_generator import (
    generate_title,
    generate_outline,
    parse_outline,
    generate_introduction,
    generate_section,
    generate_conclusion,
    generate_article_summary,
    ArticleContext,
)
from article_generator.text_processor import (
    humanize_text,
    check_grammar,
    format_article_for_wordpress,
    generate_block_notes,
    convert_wp_to_markdown,
)
from article_generator.rag_search_engine import ArticleExtractor
from article_generator.paa_handler import generate_paa_section
from article_generator.faq_handler import generate_faq_section
from article_generator.wordpress_handler import post_to_wordpress
from article_generator.image_handler import ImageConfig, get_article_images
from article_generator.youtube_handler import get_video_for_article
from article_generator.meta_handler import MetaHandler
from article_generator.external_links_handler import generate_external_links_section
from article_generator.logger import logger
from utils.api_validators import (
    validate_openai_api_key,
    validate_youtube_api_key,
    validate_serpapi_key,
    validate_wordpress_api,
    validate_duckduckgo_access,
)
from utils.rate_limiter import RateLimitConfig, initialize_rate_limiters
from article_generator.rag_retriever import WebContentRetriever
from utils.error_utils import ErrorHandler

# Initialize error handler
error_handler = ErrorHandler()
from bs4 import BeautifulSoup
from article_generator.content_generator import make_openrouter_api_call


class Generator:
    def __init__(self, config: Config, prompts: Prompts):
        """
        Initialize the article generator with configuration and prompts.

        Args:
            config (Config): Configuration object containing all settings
            prompts (Prompts): Prompts object containing all prompt templates
        """
        self.config = config
        self.prompts = prompts

        # Initialize OpenAI
        openai.api_key = self.config.openai_key

        # Initialize rate limiters
        if self.config.enable_rate_limiting:
            logger.info("Initializing rate limiters...")
            initialize_rate_limiters(
                openai_config=RateLimitConfig(
                    rpm=self.config.openai_rpm,
                    rpd=self.config.openai_rpd,
                    cooldown_period=self.config.rate_limit_cooldown,
                    enabled=self.config.enable_rate_limiting,
                ),
                serpapi_config=RateLimitConfig(
                    rpm=self.config.serpapi_rpm,
                    rpd=self.config.serpapi_rpd,
                    cooldown_period=self.config.rate_limit_cooldown,
                    enabled=self.config.enable_rate_limiting,
                ),
                duckduckgo_config=RateLimitConfig(
                    rpm=self.config.duckduckgo_rpm,
                    rpd=self.config.duckduckgo_rpd,
                    cooldown_period=self.config.rate_limit_cooldown,
                    enabled=self.config.enable_rate_limiting,
                ),
                unsplash_config=RateLimitConfig(
                    rpm=self.config.unsplash_rpm,
                    rpd=self.config.unsplash_rpd,
                    cooldown_period=self.config.rate_limit_cooldown,
                    enabled=self.config.enable_rate_limiting,
                ),
                youtube_config=RateLimitConfig(
                    rpm=self.config.youtube_rpm,
                    rpd=self.config.youtube_rpd,
                    cooldown_period=self.config.rate_limit_cooldown,
                    enabled=self.config.enable_rate_limiting,
                ),
                openverse_config=RateLimitConfig(
                    rpm=self.config.openverse_rpm,
                    rpd=self.config.openverse_rpd,
                    cooldown_period=self.config.rate_limit_cooldown,
                    enabled=self.config.enable_rate_limiting,
                ),
                pexels_config=RateLimitConfig(
                    rpm=self.config.pexels_rpm,
                    rpd=self.config.pexels_rpd,
                    cooldown_period=self.config.rate_limit_cooldown,
                    enabled=self.config.enable_rate_limiting,
                ),
                pixabay_config=RateLimitConfig(
                    rpm=self.config.pixabay_rpm,
                    rpd=self.config.pixabay_rpd,
                    cooldown_period=self.config.rate_limit_cooldown,
                    enabled=self.config.enable_rate_limiting,
                ),
                huggingface_config=RateLimitConfig(
                    rpm=self.config.huggingface_rpm,
                    rpd=self.config.huggingface_rpd,
                    cooldown_period=self.config.rate_limit_cooldown,
                    enabled=self.config.enable_rate_limiting,
                ),
            )

        # Initialize meta handler
        self.meta_handler = MetaHandler(config, openai, prompts)

        # Initialize image config with all parameters from main config
        self.image_config = ImageConfig(
            image_source=self.config.image_source,
            stock_primary_source=self.config.stock_primary_source,
            secondary_source_image=self.config.secondary_source_image,
            # AI Image settings
            huggingface_model=self.config.huggingface_model,
            huggingface_api_key=self.config.huggingface_api_key,
            # Image sources api keys
            unsplash_api_key=self.config.unsplash_api_key,
            pexels_api_key=self.config.pexels_api_key,
            pixabay_api_key=self.config.pixabay_api_key,
            giphy_api_key=self.config.giphy_api_key,
            # Image caption settings
            image_caption_instance=self.config.image_caption_instance,
            # Values
            enable_image_generation=self.config.enable_image_generation,
            max_number_of_images=self.config.max_number_of_images,
            orientation=self.config.orientation,
            order_by=self.config.order_by,
            # Add the new image parameters
            alignment=self.config.image_alignment,
            enable_image_compression=self.config.enable_image_compression,
            compression_quality=self.config.image_compression_quality,
            prevent_duplicate_images=self.config.prevent_duplicate_images,
        )

        # Initialize web content retriever for RAG only if enabled
        if hasattr(self.config, "enable_rag") and self.config.enable_rag:
            try:
                self.web_retriever = WebContentRetriever(config)
                logger.info("Web content retriever initialized for RAG")
            except Exception as e:
                logger.error(f"Failed to initialize web content retriever: {str(e)}")
                self.web_retriever = None
        else:
            logger.info(
                "RAG is disabled, skipping initialization of web content retriever"
            )
            self.web_retriever = None

        # Initialize console for rich output
        self.console = Console()

        logger.info("Generator initialized")

    def setup_environment(self) -> bool:
        """
        Set up the environment for article generation

        Returns:
            bool: True if setup was successful, False otherwise
        """
        try:
            # Create output directory if it doesn't exist
            if self.config.enable_markdown_save:
                os.makedirs(self.config.markdown_output_dir, exist_ok=True)
                logger.info(
                    f"Created output directory: {self.config.markdown_output_dir}"
                )

            # Create context save directory if it doesn't exist
            if self.config.enable_context_save:
                os.makedirs(self.config.context_save_dir, exist_ok=True)
                logger.info(
                    f"Created context save directory: {self.config.context_save_dir}"
                )

            # Validate OpenAI API key only if not using OpenRouter
            if not self.config.use_openrouter:
                if not validate_openai_api_key(self.config.openai_key):
                    return False
            else:
                # If using OpenRouter, still try to validate OpenAI API key but don't fail if invalid
                validate_openai_api_key(self.config.openai_key)
                # Log a message that we're continuing with OpenRouter
                logger.info(
                    "Using OpenRouter for API calls, continuing despite OpenAI API key status"
                )

            # Validate YouTube API key if enabled
            if self.config.enable_youtube_videos and not validate_youtube_api_key(
                self.config.youtube_api
            ):
                return False

            # Validate SerpAPI key if PAA is enabled
            if self.config.add_PAA_paragraphs_into_article:
                is_valid, key_info = validate_serpapi_key(self.config.serp_api_key)
                if not is_valid:
                    logger.warning(
                        "SerpAPI Key is invalid or exhausted! Disabling features that require SerpAPI."
                    )
                    # Automatically disable features that require SerpAPI instead of failing
                    self.config.add_PAA_paragraphs_into_article = False
                    self.config.add_external_links_into_article = False
                    logger.info("People Also Ask (PAA) feature disabled")
                    logger.info("External links feature disabled")
                    # Continue execution instead of returning False
                else:
                    logger.success(f"SerpAPI Key is VALID! Quota: {key_info}")

            if self.config.rag_article_retriever_engine == "Duckduckgo":
                logger.info(
                    f"{self.config.rag_article_retriever_engine} was chosen as the article retrieval engine"
                )
                is_ddg_valid, key_info = validate_duckduckgo_access()
                if is_ddg_valid:
                    logger.info(f"Duckduckgo is running successfully")
                else:
                    logger.info(f"Using duckduckgo regardk")
            else:
                logger.info(
                    f"{ self.config.rag_article_retriever_engine} was chosen as the article retrieval engine"
                )

            # Validate WordPress credentials if upload is enabled
            if self.config.enable_wordpress_upload:
                json_url = f"https://{self.config.website_name}/wp-json/wp/v2"
                headers = {
                    "Authorization": f"Basic {base64.b64encode(f'{self.config.Username}:{self.config.App_pass}'.encode()).decode()}"
                }
                if not validate_wordpress_api(
                    json_url,
                    headers,
                    self.config.status,
                    self.config.categories,
                    self.config.author,
                ):
                    return False

            return True
        except Exception as e:
            logger.error(f"Error in environment setup: {str(e)}")
            return False

    def clean_html(content: str) -> str:
        """Clean HTML content to avoid nested/empty <p> and ensure WP compatibility."""
        soup = BeautifulSoup(content, "html.parser")

        # Remove empty <p>
        for p in soup.find_all("p"):
            if not p.text.strip():
                p.decompose()

        # Flatten nested <p><p>...</p></p>
        for p in soup.find_all("p"):
            if p.find("p"):
                inner = ''.join(str(child) for child in p.contents)
                p.replace_with(BeautifulSoup(inner, "html.parser"))

        return str(soup)

    def build_formatting_instructions(self) -> str:
        instructions = []
        if self.config.add_bold_into_article:
            instructions.append("Use <strong> for bold keywords.")
        if self.config.add_italic_into_article:
            instructions.append("Use <em> for italic keywords.")
        if self.config.add_lists_into_articles:
            instructions.append("Use <ul><li> or <ol><li> for lists when suitable.")
        if self.config.add_tables_into_articles:
            instructions.append("If data fits a table, use <table><tr><td> HTML.")
        if self.config.enable_variable_subheadings:
            instructions.append("Vary subheadings with <h2>, <h3>, <h4> levels naturally.")
        return "\n".join(instructions)

    def generate_article(
        self,
        keyword: str,
        image_keyword: Optional[str] = None,
        author_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate a complete article for the given keyword.

        Args:
            keyword (str): Main keyword for article generation
            image_keyword (str, optional): Specific keyword for image search
            author_id (str, optional): Optional WordPress author ID to use for posting

        Returns:
            Dict[str, str]: Generated article components
        """
        logger.info(f"\n{'='*20} Generating Article for '{keyword}' {'='*20}")

        formatting_rules = self.build_formatting_instructions()

        try:
            # Create timestamp for unique file naming
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Initialize web content retriever for RAG if enabled
            web_context = None
            if hasattr(self.config, "enable_rag") and self.config.enable_rag:
                try:
                    logger.info(f"Building knowledge base for keyword: {keyword}")
                    # Only initialize if it doesn't already exist
                    if self.web_retriever is None:
                        self.web_retriever = WebContentRetriever(self.config)
                        logger.info("Web content retriever initialized for RAG")
                    # Get RAG context once for the entire article
                    web_context = self.web_retriever.get_context_for_generation(
                        keyword, num_articles=5, num_chunks=8
                    )
                    success = web_context and len(web_context) > 0
                    if not success:
                        logger.warning(f"Failed to build knowledge base for: {keyword}")
                except Exception as e:
                    logger.error(f"Error building knowledge base: {str(e)}")

            # Use the logger's progress bar
            with logger.progress_bar() as progress:
                # Initialize article context
                try:
                    context = ArticleContext(
                        config=self.config,
                        openai_engine=self.config.openai_model,
                        max_context_window_tokens=self.config.max_context_window_tokens,
                        token_padding=self.config.token_padding,
                        track_token_usage=self.config.enable_token_tracking,
                        warn_token_threshold=self.config.warn_token_threshold,
                        section_token_limit=self.config.section_token_limit,
                        paragraphs_per_section=self.config.paragraphs_per_section,
                        min_paragraph_tokens=self.config.min_paragraph_tokens,
                        max_paragraph_tokens=self.config.max_paragraph_tokens,
                        size_headings=self.config.sizeheadings,
                        size_sections=self.config.sizesections,
                        articletype=self.config.articletype,
                        articlelanguage=self.config.articlelanguage,
                        voicetone=self.config.voicetone,
                        pointofview=self.config.pointofview,
                        articleaudience=self.config.articleaudience,
                    )

                    # Verify critical attributes are present
                    required_attributes = [
                        "encoding",
                        "messages",
                        "article_parts",
                        "total_tokens",
                    ]
                    for attr in required_attributes:
                        if not hasattr(context, attr):
                            logger.error(
                                f"Critical attribute '{attr}' is missing from ArticleContext"
                            )
                            raise AttributeError(f"Missing attribute: {attr}")

                    logger.info("Article context initialized")
                except Exception as e:
                    logger.error(f"Error initializing ArticleContext: {str(e)}")
                    raise

                # Set the keyword in the context
                context.keyword = keyword

                # Add web context to the article context using the set_rag_context method
                if web_context and self.config.enable_rag:
                    # Use the set_rag_context method to properly integrate RAG content into system message
                    if hasattr(context, "set_rag_context"):
                        context.set_rag_context(web_context)
                        logger.info(
                            f"Added RAG context to article generation using set_rag_context ({len(web_context)} chars)"
                        )
                    else:
                        # Fallback to old method if set_rag_context is not available
                        rag_system_msg = {
                            "role": "system",
                            "content": f"You are an expert content writer. Use the following research information to help guide your article creation:\n\n{web_context}",
                        }
                        # Insert after the first system message
                        if len(context.messages) > 0:
                            context.messages.insert(1, rag_system_msg)
                        else:
                            context.messages.append(rag_system_msg)
                        logger.info(
                            f"Added RAG context to article generation using fallback method ({len(web_context)} chars)"
                        )

                web_context = ""

                if self.config.enable_rag_search_engine:
                    article_extractor = ArticleExtractor(
                        keyword=keyword,
                        max_search_results=12,
                        max_content_length=2000,
                        headless=True,
                    )

                    web_context_array = article_extractor.search_and_extract()
                    for web_article in web_context_array:
                        web_context += (
                            f"Title: {web_article['title']}\n"
                            f"Content: {web_article['content']}\n"
                            f"Source: {web_article['url']}\n\n"
                        )

                    messages = [
                        {
                            "role": "system",
                            "content": f"Gather all the information from the following articles and summarize them in a single place. Make sure to never ever skip any information at any cost.:\n\n{web_context}",
                        }
                    ]

                    web_context = (
                        make_openrouter_api_call(
                            messages=messages,
                            model=self.config.rag_openrouter_model,
                            api_key=context.config.openrouter_api_key,
                            site_url=context.config.openrouter_site_url,
                            site_name=context.config.openrouter_site_name,
                            max_tokens=4000,
                        )
                        .choices[0]
                        .message.content.strip()
                        .replace("```html", "")
                        .replace("```", "")
                        .strip()
                    )

                    error_handler.handle_error(Exception(web_context), severity="info")

                    rag_system_msg = {
                        "role": "system",
                        "content": f"You are an expert content writer. Make sure to follow and Use the following research information to help guide your article creation at any cost:\n\n{web_context}",
                    }
                    # Insert after the first system message
                    if len(context.messages) > 0:
                        context.messages.insert(1, rag_system_msg)
                    else:
                        context.messages.append(rag_system_msg)
                    logger.info(
                        f"Added RAG context to article generation ({len(web_context)} chars)"
                    )

                # Initialize article parts dictionary if not already present
                if not hasattr(context, "article_parts"):
                    context.article_parts = {
                        "title": "",
                        "outline": "",
                        "introduction": "",
                        "sections": [],
                        "conclusion": "",
                    }

                if context.track_token_usage:
                    logger.info(f"Token usage - Initial: {context.total_tokens_used}")

                # Generate title
                task = progress.add_task("[cyan]Generating title...", total=1)
                title = ""

                retries = 0
                max_retries = 3
                while retries < max_retries and (not title or title.strip() == ""):
                    try:
                        title = (
                            generate_title(context, keyword, self.prompts.title)
                            .strip()
                            .replace("**", "")
                            .strip()
                        )
                        retries += 1

                    except Exception as e:
                        logger.error(f"Error generating title: {str(e)}")
                        retries += 1
                        if retries < max_retries:
                            logger.info(
                                f"Retrying title generation ({retries}/{max_retries})..."
                            )
                            time.sleep(2)

                # title = generate_title(context, keyword, self.prompts.title)

                progress.update(task, completed=1)

                # Generate outline
                task = progress.add_task("[cyan]Generating outline...", total=1)
                outline = ""
                retries = 0
                max_retries = 3

                # Select appropriate prompt based on sizeheadings value
                if self.config.sizeheadings == 0:
                    # When no subsections are requested, use two-level outline
                    if self.config.enable_variable_paragraphs:
                        logger.info("Using variable two level outline mode (no subsections)")
                        outline_prompt = self.prompts.variable_two_level_outline + "\n\n" + formatting_rules
                    else:
                        logger.info("Using two level outline mode (no subsections)")
                        outline_prompt = self.prompts.two_level_outline + "\n\n" + formatting_rules
                else:
                    # When subsections are requested, use regular outline
                    if self.config.enable_variable_paragraphs:
                        logger.info("Using variable paragraphs outline mode")
                        outline_prompt = self.prompts.variable_paragraphs_outline + "\n\n" + formatting_rules
                    else:
                        logger.info("Using fixed paragraphs outline mode")
                        outline_prompt = self.prompts.fixed_outline + "\n\n" + formatting_rules

                while retries < max_retries and (not outline or outline.strip() == ""):
                    try:
                        outline = generate_outline(context, keyword, outline_prompt)
                        retries += 1
                    except Exception as e:
                        logger.error(f"Error generating outline: {str(e)}")
                        retries += 1
                        if retries < max_retries:
                            logger.info(
                                f"Retrying outline generation ({retries}/{max_retries})..."
                            )
                            time.sleep(2)
                progress.update(task, completed=1)
                parsed_sections = []
                retries = 0
                max_retries = 3
                while retries < max_retries and not parsed_sections:
                    try:
                        parsed_sections = parse_outline(outline, context)
                        retries += 1
                    except Exception as e:
                        logger.error(f"Error parsing outline: {str(e)}")
                        retries += 1
                        if retries < max_retries:
                            logger.info(
                                f"Retrying outline parsing ({retries}/{max_retries})..."
                            )
                            time.sleep(2)
                # parsed_sections = parse_outline(outline)

                # Generate introduction
                task = progress.add_task("[cyan]Generating introduction...", total=1)
                introduction = ""
                retries = 0
                max_retries = 3
                while retries < max_retries and (
                    not introduction or introduction.strip() == ""
                ):
                    try:
                        introduction = generate_introduction(
                            context, keyword, self.prompts.introduction
                        )
                        retries += 1
                    except Exception as e:
                        logger.error(f"Error generating introduction: {str(e)}")
                        retries += 1
                        if retries < max_retries:
                            logger.info(
                                f"Retrying introduction generation ({retries}/{max_retries})..."
                            )
                            time.sleep(2)
                introduction = generate_introduction(context, keyword, self.prompts.introduction + "\n\n" + formatting_rules)
                progress.update(task, completed=1)

                # Generate sections
                sections_task = progress.add_task(
                    "[cyan]Generating sections...", total=len(parsed_sections)
                )
                section_contents = []

                for section_number, section in enumerate(parsed_sections, 1):
                    progress.update(
                        sections_task,
                        description=f"[cyan]Generating section {section_number}/{len(parsed_sections)}: {section['title']}",
                    )

                    content = ""
                    retries = 0
                    max_retries = 3
                    while retries < max_retries and (
                        not content or content.strip() == ""
                    ):
                        try:
                            content = generate_section(
                                context=context,
                                heading=section["title"],
                                keyword=keyword,
                                section_number=section_number,
                                total_sections=len(parsed_sections),
                                paragraph_prompt=self.prompts.paragraph_generate + "\n\n" + formatting_rules,
                                parsed_sections=parsed_sections,
                            )
                            print(content)
                            retries += 1
                        except Exception as e:
                            logger.error(
                                f"Error generating section {section_number}: {str(e)}"
                            )
                            retries += 1
                            if retries < max_retries:
                                logger.info(
                                    f"Retrying section generation ({retries}/{max_retries})..."
                                )
                                time.sleep(2)
                    # content = generate_section(
                    #     context=context,
                    #     heading=section['title'],
                    #     keyword=keyword,
                    #     section_number=section_number,
                    #     total_sections=len(parsed_sections),
                    #     paragraph_prompt=self.prompts.paragraph_generate,
                    #     parsed_sections=parsed_sections
                    # )
                    section_contents.append(content)
                    progress.advance(sections_task)

                # Generate conclusion
                task = progress.add_task("[cyan]Generating conclusion...", total=1)
                conclusion = ""
                retries = 0
                max_retries = 3
                while retries < max_retries and (
                    not conclusion or conclusion.strip() == ""
                ):
                    try:
                        conclusion = generate_conclusion(
                            context,
                            keyword,
                            self.prompts.conclusion + "\n\n" + formatting_rules,
                            self.prompts.summarize,
                        )
                        retries += 1
                    except Exception as e:
                        logger.error(f"Error generating conclusion: {str(e)}")
                        retries += 1
                        if retries < max_retries:
                            logger.info(
                                f"Retrying conclusion generation ({retries}/{max_retries})..."
                            )
                            time.sleep(2)
                # conclusion = generate_conclusion(
                #     context, keyword,
                #     self.prompts.conclusion,
                #     self.prompts.summarize
                # )
                progress.update(task, completed=1)

                # Initialize summary as empty string
                summary = ""

                # Generate summary only if enabled in config
                if self.config.add_summary_into_article:
                    task = progress.add_task(
                        "[cyan]Generating article summary...", total=1
                    )

                    retries = 0
                    max_retries = 3
                    while retries < max_retries and (
                        not summary or summary.strip() == ""
                    ):
                        try:
                            summary = generate_article_summary(
                                context=context,
                                keyword=keyword,
                                article_dict={
                                    "title": title,
                                    "introduction": introduction,
                                    "sections": section_contents,
                                    "conclusion": conclusion + "\n\n" + formatting_rules,
                                },
                                summarize_prompt=self.prompts.summarize,
                                combine_prompt=self.prompts.summary_combine,
                            )
                            retries += 1
                        except Exception as e:
                            logger.error(f"Error generating article summary: {str(e)}")
                            retries += 1
                            if retries < max_retries:
                                logger.info(
                                    f"Retrying summary generation ({retries}/{max_retries})..."
                                )
                                time.sleep(2)
                    # summary = generate_article_summary(
                    #     context=context,
                    #     keyword=keyword,
                    #     article_dict={
                    #         'title': title,
                    #         'introduction': introduction,
                    #         'sections': section_contents,
                    #         'conclusion': conclusion
                    #     },
                    #     summarize_prompt=self.prompts.summarize
                    # )
                    progress.update(task, completed=1)

                # Create initial article dictionary
                article_dict = {
                    "title": title,
                    "introduction": introduction,
                    "sections": section_contents,
                    "conclusion": conclusion,
                    "summary": summary,
                    "paa": "",
                    "faq": "",
                    "block_notes": "",
                    "youtube_video": "",
                    "paa_section": "",
                    "faq_section": "",
                    "meta_description": "",
                    "wordpress_excerpt": "",
                    "external_links": "",
                }

                # Generate meta description and excerpt if enabled
                if self.config.enable_meta_description:
                    task = progress.add_task(
                        "[cyan]Generating meta content...", total=2
                    )

                    # Generate meta description
                    article_dict["meta_description"] = (
                        self.meta_handler.generate_meta_description(
                            keyword=keyword,
                            article_type=self.config.articletype,
                            article_audience=self.config.articleaudience,
                        )
                    )
                    progress.advance(task)

                    # Generate WordPress excerpt
                    article_dict["wordpress_excerpt"] = (
                        self.meta_handler.generate_wordpress_excerpt(
                            keyword=keyword,
                            article_type=self.config.articletype,
                            article_audience=self.config.articleaudience,
                        )
                    )
                    progress.advance(task)

                # Add YouTube video if enabled
                if self.config.enable_youtube_videos:
                    task = progress.add_task("[cyan]Adding YouTube video...", total=1)
                    video_embed = get_video_for_article(
                        self.config.youtube_api,
                        keyword,
                        output_format="markdown",  # Always get HTML iframe for markdown files
                        video_width=self.config.youtube_video_width,
                        video_height=self.config.youtube_video_height,
                    )
                    if video_embed:
                        article_dict["youtube_video"] = video_embed
                    progress.update(task, completed=1)

                # Add PAA section if enabled
                if self.config.add_PAA_paragraphs_into_article:
                    task = progress.add_task("[cyan]Generating PAA section...", total=1)
                    try:
                        if (
                            self.config.enable_rag_search_engine
                            and web_context
                            and web_context.strip() != ""
                        ):
                            article_dict["paa_section"] = generate_paa_section(
                                keyword,
                                serp_api_key=self.config.serp_api_key,
                                context=context,
                                web_context=web_context,
                                config=self.config,
                            )
                        else:
                            article_dict["paa_section"] = generate_paa_section(
                                keyword,
                                serp_api_key=self.config.serp_api_key,
                                context=context,
                                config=self.config,
                            )
                    except Exception as e:
                        logger.error(f"Error generating PAA section: {str(e)}")
                        article_dict["paa_section"] = ""
                    progress.update(task, completed=1)
                else:
                    logger.info("PAA section generation skipped (feature disabled)")
                    article_dict["paa_section"] = ""

                # Add FAQ section if enabled
                if self.config.add_faq_into_article:
                    task = progress.add_task("[cyan]Generating FAQ section...", total=1)
                    try:
                        logger.info("Generating FAQ section...")
                        faq_section = generate_faq_section(
                            context=context,
                            keyword=keyword,
                            faq_prompt=self.prompts.faqs,
                            openai_engine=self.config.openai_model,
                            enable_token_tracking=self.config.enable_token_tracking,
                            track_token_usage=self.config.enable_progress_display,
                            web_context=web_context,
                        )

                        # Make sure we have content before adding it
                        if faq_section and faq_section.strip():
                            article_dict["faq_section"] = faq_section
                            logger.success("FAQ section generated successfully")
                        else:
                            logger.warning(
                                "FAQ section generation returned empty content"
                            )
                    except Exception as e:
                        logger.error(f"Error generating FAQ section: {str(e)}")
                    finally:
                        progress.update(task, completed=1)

                # Add block notes if enabled
                if self.config.add_blocknote_into_article:
                    task = progress.add_task(
                        "[cyan]Generating key takeaways...", total=1
                    )
                    article_dict["block_notes"] = generate_block_notes(
                        context=context,
                        article_content=article_dict,
                        blocknote_prompt=self.prompts.blocknote,
                        combine_prompt=self.prompts.blocknotes_combine,
                        engine=self.config.openai_model,
                        enable_token_tracking=self.config.enable_token_tracking,
                        track_token_usage=self.config.enable_token_tracking,
                    )
                    progress.update(task, completed=1)

                # Add external links if enabled
                if self.config.add_external_links_into_article:
                    task = progress.add_task(
                        "[cyan]Generating external links...", total=1
                    )
                    logger.debug(
                        f"Starting external links generation for keyword: {keyword}"
                    )

                    try:
                        # Check the SerpAPI key to make sure it's valid
                        logger.debug(
                            f"Using SerpAPI key: {self.config.serp_api_key[:5]}... (truncated)"
                        )

                        # Try with more specific search query to improve results
                        search_keyword = f"{keyword} tips advice resources"
                        logger.debug(f"Using enhanced search query: '{search_keyword}'")

                        # Always use 'wordpress' format for external links when WordPress upload is enabled
                        external_links = generate_external_links_section(
                            search_keyword,  # Use enhanced search query
                            serp_api_key=self.config.serp_api_key,
                            output_format="wordpress",  # Always use WordPress format for consistency
                        )

                        # If first attempt failed, try with just the keyword
                        if not external_links:
                            logger.debug(
                                f"First attempt failed. Trying with original keyword: '{keyword}'"
                            )
                            external_links = generate_external_links_section(
                                keyword,
                                serp_api_key=self.config.serp_api_key,
                                output_format="wordpress",
                            )

                        # If still no results, try with a different query format
                        if not external_links:
                            fallback_keyword = f"best {keyword} guide"
                            logger.debug(
                                f"Second attempt failed. Trying with fallback keyword: '{fallback_keyword}'"
                            )
                            external_links = generate_external_links_section(
                                fallback_keyword,
                                serp_api_key=self.config.serp_api_key,
                                output_format="wordpress",
                            )

                        # Check if external_links has content
                        if external_links and external_links.strip():
                            logger.success(
                                f"External links generated successfully (length: {len(external_links)})"
                            )
                            logger.debug(
                                f"External links content: {external_links[:200]}..."
                            )
                            article_dict["external_links"] = external_links
                        else:
                            logger.warning(
                                "All attempts to generate external links returned empty content"
                            )
                            # Create a placeholder message for external links
                            logger.debug("Creating placeholder external links section")
                            article_dict["external_links"] = (
                                self._create_placeholder_external_links()
                            )

                        # Debug output
                        logger.debug(
                            f"article_dict keys after external links: {list(article_dict.keys())}"
                        )
                        logger.debug(
                            f"External links content exists: {'external_links' in article_dict}"
                        )
                        logger.debug(
                            f"External links content length: {len(article_dict.get('external_links', ''))}"
                        )
                    except Exception as e:
                        logger.error(f"Error generating external links: {str(e)}")
                        # Create a placeholder message for external links
                        article_dict["external_links"] = (
                            self._create_placeholder_external_links()
                        )

                    # Always update the progress task
                    progress.update(task, completed=1)
                else:
                    logger.info("External links generation skipped (feature disabled)")
                    article_dict["external_links"] = ""

                # Post-processing
                if self.config.enable_text_humanization:
                    task = progress.add_task("[cyan]Humanizing text...", total=6)
                    article_dict = self._humanize_text(
                        article_dict, context, progress, task
                    )

                if self.config.enable_grammar_check:
                    task = progress.add_task("[cyan]Checking grammar...", total=6)
                    article_dict = self._check_grammar(
                        article_dict, context, progress, task
                    )

                # Get images if enabled - moved here to be available for both markdown and wordpress
                feature_image = None
                body_images = None
                if self.config.add_image_into_article:
                    task = progress.add_task("[cyan]Generating images...", total=1)
                    feature_image, body_images = get_article_images(
                        image_keyword or keyword,
                        self.image_config,
                        save_dir="images",
                        timestamp=timestamp,  # Pass timestamp for unique image names
                    )
                    progress.update(task, completed=1)

                # Save and upload
                if self.config.enable_markdown_save:
                    task = progress.add_task("[cyan]Saving as markdown...", total=1)
                    self._save_as_markdown(
                        article_dict, keyword, feature_image, body_images, timestamp
                    )
                    progress.update(task, completed=1)

                # Upload to WordPress if enabled - moved here to be available for both markdown and wordpress
                if self.config.enable_wordpress_upload:
                    task = progress.add_task("[cyan]Uploading to WordPress...", total=1)
                    self._upload_to_wordpress(
                        article_dict, keyword, feature_image, body_images, author_id
                    )
                    progress.update(task, completed=1)

                # Save the article context if enabled
                if self.config.enable_context_save:
                    context.save_to_file()

                return article_dict

        except Exception as e:
            logger.error(f"Error generating article: {str(e)}")
            raise

    def _check_grammar(
        self,
        article_dict: Dict[str, str],
        context: ArticleContext,
        progress: Optional[Progress] = None,
        task: Optional[TaskID] = None,
    ) -> Dict[str, str]:
        """Check grammar for all article components with support for subsections."""
        try:
            # Check title grammar
            article_dict["title"] = check_grammar(
                context,
                article_dict["title"],
                self.prompts.grammar,
                engine=self.config.openai_model,
                enable_token_tracking=self.config.enable_token_tracking,
                track_token_usage=self.config.enable_token_tracking,
                content_type="Title",
            )
            if progress and task:
                progress.advance(task)

            # Check introduction grammar
            article_dict["introduction"] = check_grammar(
                context,
                article_dict["introduction"],
                self.prompts.grammar,
                engine=self.config.openai_model,
                enable_token_tracking=self.config.enable_token_tracking,
                track_token_usage=self.config.enable_token_tracking,
                content_type="Introduction",
            )
            if progress and task:
                progress.advance(task)

            # Check sections grammar with subsection support
            for i, section in enumerate(article_dict["sections"]):
                logger.info(
                    f"Processing grammar for section {i+1} with subsection support"
                )

                # Parse the section structure
                structured_content = self._parse_content_structure(section)

                # Process the structured content
                processed_section = self._process_structured_content(
                    structured_content=structured_content,
                    context=context,
                    prompt=self.prompts.grammar,
                    engine=self.config.openai_model,
                    enable_token_tracking=self.config.enable_token_tracking,
                    track_token_usage=self.config.enable_token_tracking,
                    processing_function=check_grammar,
                    content_type=f"Section {i+1}",
                )

                article_dict["sections"][i] = processed_section

            if progress and task:
                progress.advance(task)

            # Check conclusion grammar
            article_dict["conclusion"] = check_grammar(
                context,
                article_dict["conclusion"],
                self.prompts.grammar,
                engine=self.config.openai_model,
                enable_token_tracking=self.config.enable_token_tracking,
                track_token_usage=self.config.enable_token_tracking,
                content_type="Conclusion",
            )
            if progress and task:
                progress.advance(task)

            # Check FAQ section grammar if present - preserve Q:/A: structure
            if article_dict.get("faq_section"):
                faq_content = article_dict["faq_section"]
                faq_lines = faq_content.split("\n")
                grammar_checked_lines = []

                for line in faq_lines:
                    line = line.strip()
                    if not line:
                        grammar_checked_lines.append("")
                        continue

                    # Preserve Q: and A: markers, only grammar check content
                    if line.startswith("Q:") or line.startswith("A:"):
                        marker = line[:2]  # Get Q: or A:
                        content = line[2:].strip()  # Get content after marker

                        if content:  # Grammar check both questions and answers
                            try:
                                grammar_checked_content = check_grammar(
                                    context,
                                    content,
                                    self.prompts.grammar,
                                    engine=self.config.openai_model,
                                    enable_token_tracking=False,  # Avoid token spam
                                    track_token_usage=False,
                                    content_type="FAQ Content",
                                )
                                grammar_checked_lines.append(
                                    f"{marker} {grammar_checked_content}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Error grammar checking FAQ content: {str(e)}"
                                )
                                grammar_checked_lines.append(
                                    line
                                )  # Keep original on error
                        else:
                            grammar_checked_lines.append(line)
                    else:
                        # For any other content, preserve as-is
                        grammar_checked_lines.append(line)

                article_dict["faq_section"] = "\n".join(grammar_checked_lines)
                logger.info(
                    "FAQ section grammar checked while preserving Q:/A: structure"
                )
            if progress and task:
                progress.advance(task)

            # Check PAA section grammar if present - preserve markdown structure with subsections
            if article_dict.get("paa_section"):
                logger.info("Processing PAA section grammar with subsection support")

                # Parse the PAA section structure
                structured_content = self._parse_content_structure(
                    article_dict["paa_section"]
                )

                # Process the structured content
                processed_paa = self._process_structured_content(
                    structured_content=structured_content,
                    context=context,
                    prompt=self.prompts.grammar,
                    engine=self.config.openai_model,
                    enable_token_tracking=False,  # Avoid token spam
                    track_token_usage=False,
                    processing_function=check_grammar,
                    content_type="PAA",
                )

                article_dict["paa_section"] = processed_paa
                logger.info(
                    "PAA section grammar checked while preserving markdown structure with subsections"
                )

            if progress and task:
                progress.advance(task)

            return article_dict

        except Exception as e:
            logger.error(f"Error in grammar checking: {str(e)}")
            return article_dict

    def _humanize_text(
        self,
        article_dict: Dict[str, str],
        context: ArticleContext,
        progress: Progress,
        task: int,
    ) -> Dict[str, str]:
        """Helper method to humanize article components with support for subsections."""
        # Humanize title
        article_dict["title"] = humanize_text(
            context,
            article_dict["title"],
            self.prompts.humanize,
            engine=self.config.openai_model,
            enable_token_tracking=self.config.enable_token_tracking,
            track_token_usage=self.config.enable_token_tracking,
            content_type="title",
        )
        progress.advance(task)

        # Humanize introduction
        article_dict["introduction"] = humanize_text(
            context,
            article_dict["introduction"],
            self.prompts.humanize,
            engine=self.config.openai_model,
            enable_token_tracking=self.config.enable_token_tracking,
            track_token_usage=self.config.enable_token_tracking,
            content_type="introduction",
        )
        progress.advance(task)

        # Humanize sections with subsection support
        for i, section in enumerate(article_dict["sections"]):
            logger.info(f"Humanizing section {i+1} with subsection support")

            # Parse the section structure
            structured_content = self._parse_content_structure(section)

            # Process the structured content
            processed_section = self._process_structured_content(
                structured_content=structured_content,
                context=context,
                prompt=self.prompts.humanize,
                engine=self.config.openai_model,
                enable_token_tracking=self.config.enable_token_tracking,
                track_token_usage=self.config.enable_token_tracking,
                processing_function=humanize_text,
                content_type=f"Section {i+1}",
            )

            article_dict["sections"][i] = processed_section

        progress.advance(task)

        # Humanize conclusion
        article_dict["conclusion"] = humanize_text(
            context,
            article_dict["conclusion"],
            self.prompts.humanize,
            engine=self.config.openai_model,
            enable_token_tracking=self.config.enable_token_tracking,
            track_token_usage=self.config.enable_token_tracking,
            content_type="conclusion",
        )
        progress.advance(task)

        # Humanize FAQ if present - preserve Q:/A: structure
        if article_dict["faq_section"]:
            faq_content = article_dict["faq_section"]
            faq_lines = faq_content.split("\n")
            humanized_lines = []

            for line in faq_lines:
                line = line.strip()
                if not line:
                    humanized_lines.append("")
                    continue

                # Preserve Q: and A: markers
                if line.startswith("Q:") or line.startswith("A:"):
                    marker = line[:2]  # Get Q: or A:
                    content = line[2:].strip()  # Get content after marker

                    if (
                        content and marker == "A:"
                    ):  # Only humanize answers, not questions
                        try:
                            humanized_content = humanize_text(
                                context,
                                content,
                                self.prompts.humanize,
                                engine=self.config.openai_model,
                                enable_token_tracking=False,  # Avoid token spam
                                track_token_usage=False,
                            )
                            humanized_lines.append(f"{marker} {humanized_content}")
                        except Exception as e:
                            logger.warning(f"Error humanizing FAQ answer: {str(e)}")
                            humanized_lines.append(line)  # Keep original on error
                    else:
                        # Keep questions and empty content as-is
                        humanized_lines.append(line)
                else:
                    # For any other content, preserve as-is
                    humanized_lines.append(line)

            article_dict["faq_section"] = "\n".join(humanized_lines)
            logger.info("FAQ section humanized while preserving Q:/A: structure")
        progress.advance(task)

        # Humanize PAA if present - with subsection support
        if article_dict["paa_section"]:
            logger.info("Humanizing PAA section with subsection support")

            # Parse the PAA section structure
            structured_content = self._parse_content_structure(
                article_dict["paa_section"]
            )

            # Process the structured content
            processed_paa = self._process_structured_content(
                structured_content=structured_content,
                context=context,
                prompt=self.prompts.humanize,
                engine=self.config.openai_model,
                enable_token_tracking=False,  # Avoid token spam
                track_token_usage=False,
                processing_function=humanize_text,
                content_type="PAA",
            )

            article_dict["paa_section"] = processed_paa
            logger.info(
                "PAA section humanized while preserving markdown structure with subsections"
            )

        progress.advance(task)

        return article_dict

    def _save_as_markdown(
        self,
        article_dict: Dict[str, str],
        keyword: str,
        feature_image: Optional[Dict[str, str]] = None,
        body_images: Optional[List[Dict[str, str]]] = None,
        timestamp: str = None,
    ) -> str:
        """Helper method to save article as markdown."""
        # Create output directory if it doesn't exist
        os.makedirs(self.config.markdown_output_dir, exist_ok=True)

        # Create filename from keyword and timestamp
        if timestamp is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{keyword.replace(' ', '_')}_{timestamp}.md"
        filepath = os.path.join(self.config.markdown_output_dir, filename)

        # Format article as markdown
        markdown_content = f"# {article_dict['title']}\n\n"

        # Add feature image if available
        if feature_image and self.config.add_image_into_article:
            markdown_content += f"![{feature_image['alt']}]({feature_image['url']})\n"
            markdown_content += f"*{feature_image['caption']}*\n\n"

        # Add summary if enabled
        if self.config.add_summary_into_article and article_dict.get("summary"):
            markdown_content += "## Summary\n\n"
            markdown_content += f"{article_dict['summary']}\n\n"
            markdown_content += "---\n\n"

        # Add introduction
        markdown_content += (
            convert_wp_to_markdown(article_dict["introduction"]) + "\n\n"
        )

        # Add YouTube video after introduction if configured
        if (
            self.config.enable_youtube_videos
            and article_dict.get("youtube_video")
            and self.config.youtube_position == "after_introduction"
        ):
            markdown_content += article_dict["youtube_video"] + "\n\n"

        # Add main sections with images interspersed if available
        for i, section in enumerate(article_dict["sections"]):
            # Add YouTube video after first section if configured
            if (
                i == 0
                and self.config.enable_youtube_videos
                and article_dict.get("youtube_video")
                and self.config.youtube_position == "after_first_section"
            ):
                markdown_content += article_dict["youtube_video"] + "\n\n"

            markdown_content += convert_wp_to_markdown(section) + "\n\n"

            if (
                body_images
                and i < len(body_images)
                and self.config.add_image_into_article
            ):
                image = body_images[i]
                markdown_content += f"![{image['alt']}]({image['url']})\n"
                markdown_content += f"*{image['caption']}*\n\n"

        # Add conclusion
        markdown_content += "## Conclusion\n\n"
        markdown_content += convert_wp_to_markdown(article_dict["conclusion"]) + "\n\n"

        # Add YouTube video at end if configured
        if (
            self.config.enable_youtube_videos
            and article_dict.get("youtube_video")
            and self.config.youtube_position == "end"
        ):
            markdown_content += article_dict["youtube_video"] + "\n\n"

        # Add block notes if present
        if article_dict.get("block_notes"):
            markdown_content += "## Key Takeaways\n\n"
            # Clean up block notes content
            block_notes = (
                article_dict["block_notes"].replace("## Key Takeaways\n\n", "").strip()
            )

            # Format as a blockquote with additional styling
            markdown_content += '<blockquote class="key-takeaways-block">\n'
            markdown_content += f"{block_notes}\n"
            markdown_content += "</blockquote>\n\n"

        # Add FAQ section if present
        if article_dict.get("faq_section"):
            markdown_content += "## Frequently Asked Questions\n\n"
            markdown_content += (
                convert_wp_to_markdown(article_dict["faq_section"]) + "\n\n"
            )

        # Add PAA section if present
        if article_dict.get("paa_section"):
            markdown_content += "## People Also Ask\n\n"
            markdown_content += (
                convert_wp_to_markdown(article_dict["paa_section"]) + "\n\n"
            )

        # Add external links if present
        if (
            article_dict.get("external_links")
            and article_dict["external_links"].strip()
        ):
            logger.debug(
                f"Adding external links to markdown (length: {len(article_dict['external_links'])})"
            )

            # If the external links are in WordPress format, convert them to markdown
            if "<!-- wp:" in article_dict["external_links"]:
                logger.debug("Converting WordPress format external links to markdown")

                # Extract the links from the WordPress format
                links_md = []

                # Simple extraction of links from WordPress format
                import re

                links = re.findall(
                    r'<a href="([^"]+)"[^>]*>([^<]+)</a>',
                    article_dict["external_links"],
                )

                if links:
                    links_md.append("## External Resources\n")
                    for url, title in links:
                        links_md.append(f"- [{title}]({url})")

                    markdown_content += "\n".join(links_md) + "\n\n"
                    logger.debug("Successfully added external links in markdown format")
                else:
                    logger.warning("Could not extract links from WordPress format")
            else:
                # If already in markdown format, add directly
                markdown_content += article_dict["external_links"] + "\n\n"
                logger.debug(
                    "Added external links that were already in markdown format"
                )

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        return filepath

    def _create_placeholder_external_links(self) -> str:
        """Create a placeholder for external links when SerpAPI fails."""
        placeholder_links = [
            "<!-- wp:heading -->",
            "<h2>External Resources</h2>",
            "<!-- /wp:heading -->",
            "<!-- wp:paragraph -->",
            "<p>Additional resources related to this topic will be updated soon.</p>",
            "<!-- /wp:paragraph -->",
        ]
        logger.debug("Created placeholder external links")
        return "\n".join(placeholder_links)

    def _upload_to_wordpress(
        self,
        article_dict: Dict[str, str],
        keyword: str,
        feature_image: Optional[Dict[str, str]] = None,
        body_images: Optional[List[Dict[str, str]]] = None,
        author_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """Upload the article to WordPress.

        Args:
            article_dict: Dictionary containing article components
            keyword: Main keyword for the article
            feature_image: Feature image data
            body_images: List of body image data
            author_id: Optional WordPress author ID to override the default

        Returns:
            Optional[Dict]: WordPress API response or None if failed
        """
        try:
            # Check if external links are present in article_dict before formatting
            has_external_links = (
                "external_links" in article_dict
                and article_dict["external_links"].strip()
            )
            logger.debug(
                f"External links present in article_dict before formatting: {has_external_links}"
            )
            if has_external_links:
                logger.debug(
                    f"External links length before formatting: {len(article_dict['external_links'])}"
                )
                logger.debug(
                    f"External links start with: {article_dict['external_links'][:100]}..."
                )

            # Format article for WordPress
            wp_content = format_article_for_wordpress(
                self.config,
                article_dict,
                youtube_position=self.config.youtube_position,
                body_images=body_images,
                add_summary=self.config.add_summary_into_article,  # Pass summary config
            )

            # Check if external links appear to be in the formatted content
            external_links_in_content = "External Resources" in wp_content
            logger.debug(
                f"External links heading present in formatted WordPress content: {external_links_in_content}"
            )

            # Prepare article data - remove any quotes from title
            article_data = {
                "title": article_dict["title"]
                .replace('"', "")
                .replace("'", "")
                .strip(),
                "content": wp_content,
            }

            # Get feature image URL if available
            feature_image_url = None
            if feature_image and isinstance(feature_image, dict):
                feature_image_url = feature_image.get("url")
                logger.debug(f"Using feature image URL: {feature_image_url}")

            # Get body image URLs if available
            body_image_urls = None
            if body_images:
                body_image_urls = [
                    img.get("url")
                    for img in body_images
                    if isinstance(img, dict) and img.get("url")
                ]
                if body_image_urls:
                    logger.debug(f"Found {len(body_image_urls)} body image URLs")

            # Use provided author_id if available, otherwise use the default from config
            wordpress_author = (
                author_id if author_id is not None else self.config.author
            )
            logger.debug(f"Using WordPress author ID: {wordpress_author}")

            # Post to WordPress
            response = post_to_wordpress(
                config=self.image_config,
                website_name=self.config.website_name,
                Username=self.config.Username,
                App_pass=self.config.App_pass,
                categories=self.config.categories,
                author=wordpress_author,
                status=self.config.status,
                article=article_data,
                feature_image=feature_image_url,
                body_images=body_image_urls,
                meta_description=article_dict.get("meta_description"),
                wordpress_excerpt=article_dict.get("wordpress_excerpt"),
                keyword=keyword,  # Pass the keyword for permalink slug
            )

            return response

        except Exception as e:
            logger.error(f"Error uploading to WordPress: {str(e)}")
            return None

    def _parse_content_structure(self, content: str) -> List[Dict]:
        """
        Parse content into structured format that handles both sections and subsections.

        Returns:
            List of dicts with 'type', 'level', 'content', and 'is_heading' keys
        """
        lines = content.split("\n")
        structured_content = []
        current_block = []

        for line in lines:
            stripped_line = line.strip()

            # Check if it's a heading (# ## ### etc.)
            if stripped_line.startswith("#"):
                # Save previous block if it exists
                if current_block:
                    block_content = "\n".join(current_block).strip()
                    if block_content:
                        structured_content.append(
                            {
                                "type": "content",
                                "level": 0,
                                "content": block_content,
                                "is_heading": False,
                            }
                        )
                    current_block = []

                # Determine heading level
                heading_level = len(stripped_line) - len(stripped_line.lstrip("#"))
                heading_text = stripped_line.lstrip("# ").strip()

                structured_content.append(
                    {
                        "type": "heading",
                        "level": heading_level,
                        "content": stripped_line,
                        "is_heading": True,
                        "heading_text": heading_text,
                    }
                )
            else:
                # Regular content line
                current_block.append(line)

        # Don't forget the last block
        if current_block:
            block_content = "\n".join(current_block).strip()
            if block_content:
                structured_content.append(
                    {
                        "type": "content",
                        "level": 0,
                        "content": block_content,
                        "is_heading": False,
                    }
                )

        return structured_content

    def _process_structured_content(
        self,
        structured_content: List[Dict],
        context,
        prompt,
        engine: str,
        enable_token_tracking: bool,
        track_token_usage: bool,
        processing_function,
        content_type: str = "content",
    ) -> str:
        """
        Process structured content while preserving hierarchy.

        Args:
            structured_content: List of content blocks from _parse_content_structure
            context: Article context
            prompt: Processing prompt
            engine: OpenAI engine
            enable_token_tracking: Whether to track tokens
            track_token_usage: Whether to track usage
            processing_function: Function to use for processing (humanize_text or check_grammar)
            content_type: Type of content being processed

        Returns:
            Processed content as string
        """
        processed_blocks = []

        for block in structured_content:
            if block["is_heading"]:
                # Preserve headings as-is
                processed_blocks.append(block["content"])
            else:
                # Process content blocks
                try:
                    # Split content block into paragraphs for better processing
                    paragraphs = block["content"].split("\n\n")
                    processed_paragraphs = []

                    for paragraph in paragraphs:
                        paragraph = paragraph.strip()
                        if not paragraph:
                            processed_paragraphs.append("")
                            continue

                        # Skip any remaining headings that might be in content
                        if paragraph.startswith("#"):
                            processed_paragraphs.append(paragraph)
                            continue

                        # Process the paragraph
                        processed_paragraph = processing_function(
                            context,
                            paragraph,
                            prompt,
                            engine=engine,
                            enable_token_tracking=enable_token_tracking,
                            track_token_usage=track_token_usage,
                            content_type=f"{content_type} paragraph",
                        )
                        processed_paragraphs.append(processed_paragraph)

                    # Rejoin paragraphs
                    processed_content = "\n\n".join(processed_paragraphs)
                    processed_blocks.append(processed_content)

                except Exception as e:
                    logger.warning(f"Error processing {content_type} block: {str(e)}")
                    processed_blocks.append(block["content"])  # Keep original on error

        # Rejoin all blocks with proper spacing
        result = []
        for i, block_content in enumerate(processed_blocks):
            result.append(block_content)

            # Add proper spacing between blocks
            if i < len(processed_blocks) - 1:
                # Check if current block is a heading or next block is a heading
                current_is_heading = block_content.strip().startswith("#")
                next_is_heading = processed_blocks[i + 1].strip().startswith("#")

                if current_is_heading or next_is_heading:
                    result.append("")  # Single newline after/before headings
                else:
                    result.append("")  # Single newline between content blocks

        return "\n".join(result)
