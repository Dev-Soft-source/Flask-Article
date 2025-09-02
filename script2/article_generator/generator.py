# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from typing import Dict, List, Optional, Any
import os
import time
from utils.rich_provider import provider
import traceback
import openai
from datetime import datetime
import json
import re
import random
import csv

from config import Config
from utils.prompts_config import Prompts
from utils.csv_utils import CSVProcessor
from utils.api_utils import APIValidator
from utils.text_utils import TextProcessor
from utils.rate_limiter import openai_rate_limiter, initialize_rate_limiters, RateLimitConfig
from utils.ai_utils import generate_completion, make_openrouter_api_call
from utils.error_utils import ErrorHandler, format_error_message

from article_generator.content_generator import ContentGenerator, ArticleContext
from article_generator.image_handler import ImageConfig, process_feature_image, process_body_image, fetch_image,image_randomizer,randomize_image_selection
from article_generator.paa_handler import PAAHandler
from article_generator.wordpress_handler import WordPressHandler, post_to_wordpress
from article_generator.meta_handler import MetaHandler
from article_generator.text_processor import format_article_for_wordpress, generate_block_notes
from article_generator.youtube_handler import YouTubeHandler
from article_generator.rag_retriever import WebContentRetriever
from article_generator.logger import logger
from article_generator.rag_search_engine import ArticleExtractor
from bs4 import BeautifulSoup

# Global error handler
error_handler = ErrorHandler(show_traceback=True)

def log_error(e: Exception, context: Dict[str, Any] = None, severity: str = "error"):
    """Centralized error logging function that uses error handler and logger
    
    Args:
        e: The exception
        context: Additional context for the error
        severity: Error severity level
    """
    context = context or {}
    error_handler.handle_error(e, context, severity=severity)
    logger.error(f"{e.__class__.__name__}: {str(e)}", show_traceback=True)

class Generator:
    """Main generator class for article generation."""

    def __init__(self, config: Config, prompts: Prompts):
        """Initialize the generator with configuration and prompts."""
        try:
            provider.debug("Initializing Generator...")
            self.config = config
            self.prompts = prompts

            # Initialize rate limiters if not already done
            if hasattr(self.config, 'enable_rate_limiting') and self.config.enable_rate_limiting:
                # Check if rate limiters need initialization
                if openai_rate_limiter is None:
                    provider.info("Initializing rate limiters in Generator...")
                    initialize_rate_limiters(
                        openai_config=RateLimitConfig(
                            rpm=getattr(self.config, 'openai_rpm', 60),
                            rpd=getattr(self.config, 'openai_rpd', 10000),
                            enabled=True
                        )
                    )

            # Initialize OpenAI client
            self.openai_client = openai.OpenAI(api_key=config.openai_key)

            # Log API provider being used
            if config.use_openrouter and config.openrouter_api_key:
                provider.info(f"Using OpenRouter for API calls with site: {config.openrouter_site_name}")
            else:
                provider.info(f"Using OpenAI direct API with model: {config.openai_model}")

            # Initialize components
            self.content_generator = ContentGenerator(config, prompts)
            self.paa_handler = PAAHandler(config)
            self.text_processor = TextProcessor(config)
            self.wordpress_handler = WordPressHandler(config)
            self.meta_handler = MetaHandler(config, self.openai_client, prompts)
            self.youtube_handler = YouTubeHandler(config) if config.youtube_api_key else None
            self.csv_processor = None  # Will be initialized in process_csv

            # Initialize RAG if enabled
            if config.enable_rag:
                logger.info("Initializing RAG system in Generator...")
                self.rag_retriever = WebContentRetriever(config)
                logger.success("RAG system initialized successfully in Generator")
            else:
                self.rag_retriever = None
                logger.info("RAG system disabled in Generator")

            # Initialize image config with all parameters from main config
            self.image_config = ImageConfig(
                enable_image_generation=config.enable_image_generation,
                max_number_of_images=config.max_number_of_images,
                orientation=config.orientation,
                order_by=config.order_by,
                
                
                image_source=self.config.image_source,
                stock_primary_source=self.config.stock_primary_source,
                secondary_source_image=self.config.secondary_source_image,
                image_api=self.config.image_api,
                
                # AI Image settings
                huggingface_model=self.config.huggingface_model,
                huggingface_api_key=self.config.huggingface_api_key,
                
                # Image caption settings
                image_caption_instance=self.config.image_caption_instance,
                
                # Image sources api keys
                unsplash_api_key=self.config.unsplash_api_key,
                pexels_api_key=self.config.pexels_api_key,
                pixabay_api_key=self.config.pixabay_api_key,
                giphy_api_key=self.config.giphy_api_key,
                
                # Add the new image parameters
                alignment=config.image_alignment,
                enable_image_compression=config.enable_image_compression,
                compression_quality=config.image_compression_quality,
                prevent_duplicate_images=config.prevent_duplicate_images
            )

            provider.success("Generator initialized successfully")
        except Exception as e:
            provider.error(f"Error initializing Generator: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    def validate_apis(self) -> bool:
        """Validate API keys and connections."""
        try:
            provider.info("Validating APIs...")

            # Validate OpenAI API key only if not using OpenRouter
            if not self.config.use_openrouter:
                if not APIValidator.validate_openai_key(self.config.openai_key):
                    provider.error("OpenAI API validation failed")
                    return False
            else:
                # If using OpenRouter, still try to validate OpenAI API key but don't fail if invalid
                APIValidator.validate_openai_key(self.config.openai_key)
                # Log a message that we're continuing with OpenRouter
                provider.info("Using OpenRouter for API calls, continuing despite OpenAI API key status")

            # Validate SerpAPI if PAA or RAG is enabled
            if self.config.add_paa_paragraphs_into_article or self.config.enable_rag:
                logger.info("Validating SerpAPI key for RAG/PAA functionality")
                validation_result = APIValidator.validate_serp_api_key(self.config.serp_api_key)

                if not validation_result:
                    # Instead of failing, disable features that require SerpAPI
                    logger.warning("SerpAPI validation failed.")
                    provider.warning("SerpAPI key is invalid or quota exhausted! Continuing with PAA features disabled.")

                    # Disable RAG and PAA features
                    if self.config.enable_rag:
                        if self.config.enable_rag_search_engine == "Google":
                            provider.info("Switch to duckduckgo as Rag retriever Engine")
                            self.config.enable_rag_search_engine = "Duckduckgo"
                           

                    if self.config.add_paa_paragraphs_into_article:
                        self.config.add_paa_paragraphs_into_article = False
                        logger.info("People Also Ask (PAA) feature has been disabled")

                    # Continue execution rather than failing
                else:
                    logger.success("SerpAPI validation successful")
            
            if self.config.rag_article_retriever_engine == "Duckduckgo":
                logger.info(f'{self.config.rag_article_retriever_engine} was chosen as the article retrieval engine')
                is_ddg_valid, key_info = APIValidator.validate_duckduckgo_access()
                if is_ddg_valid:
                    logger.info(f"Duckduckgo is running successfully")
                
            else:
                logger.info(f'{self.config.rag_article_retriever_engine} was chosen as the article retrieval engine')

            # Validate YouTube API if enabled
            if self.config.add_youtube_video:
                if not self.config.youtube_api_key:
                    provider.error("YouTube API key is missing")
                    return False
                if not APIValidator.validate_youtube_api_key(self.config.youtube_api_key):
                    provider.error("YouTube API validation failed")
                    return False

            # Validate Unsplash API if image generation is enabled
            if self.config.enable_image_generation:
                if not self.config.image_api:
                    provider.error("Image API key is missing")
                    return False

            # Validate WordPress if upload is enabled
            if self.config.enable_wordpress_upload:
                if not self.wordpress_handler.validate_connection():
                    provider.error("WordPress validation failed")
                    return False

            provider.success("API validation completed. Some features may have been disabled due to API limitations.")
            return True

        except Exception as e:
            provider.error(f"API validation error: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return False

    def process_csv(self, file_path: str) -> bool:
        """Process CSV file and validate its structure."""
        try:
            provider.info(f"Processing CSV file: {file_path}")

            # Initialize CSV processor with the file path
            self.csv_processor = CSVProcessor(file_path, self.config)

            # Validate and process the file
            success, message = self.csv_processor.validate_and_process()

            if success:
                # Display CSV structure for debugging
                self.csv_processor.display_csv_structure()
                provider.success(message)

                # Display additional information about the articles to be generated
                total_articles = self.csv_processor.get_total_articles()

                # Check if we have subtitle columns (structured mode)
                if not self.csv_processor.unified_processor.simple_mode:
                    provider.info(f"\nWill now create {total_articles} articles with the specified subheadings.")
                    provider.info("Each article will include all the subheadings defined in your CSV file.")
                else:
                    provider.info(f"\nWill now create {total_articles} articles with the main keywords.")
                    provider.info("Since no subheadings were provided, the articles will be generated with auto-created sections.")
            else:
                provider.error(message)
                # Display CSV template as a guide
                self.csv_processor.display_csv_template()
                provider.error("\nPlease correct your input document and try again.")

            return success

        except Exception as e:
            provider.error(f"Error processing CSV: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return False

    def create_article_context(self, keyword: str, source_data: Dict = None) -> ArticleContext:
        """Create a new article context for generation."""
        try:
            provider.debug("Creating article context...")

            # Initialize article context with configuration and prompts
            context = ArticleContext(
                config=self.config,
                prompts=self.prompts
            )

            # Set the context in the content generator
            self.content_generator.context = context

            # Initialize web_context as empty string
            web_context = ""

            # Only perform RAG search if both enable_rag and enable_rag_search_engine are True
            if self.config.enable_rag and self.config.enable_rag_search_engine:
                # Perform a search using the keyword
                article_extractor = ArticleExtractor(
                keyword=keyword,
                max_search_results=12,
                max_content_length=2000,
                headless=True
                )

                logger.info("Firing RAG Search")

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
                        "content": f"Gather all the information from the following articles and summarize them in a single place. Make sure to never ever skip any information at any cost.:\n\n{web_context}"
                    }
                ]

                # Get generation parameters from config or use defaults
                rag_temperature = getattr(self.config, 'rag_temperature', 0.7)
                rag_top_p = getattr(self.config, 'rag_top_p', 1.0)
                rag_frequency_penalty = getattr(self.config, 'rag_frequency_penalty', 0.0)
                rag_presence_penalty = getattr(self.config, 'rag_presence_penalty', 0.0)

                # Get seed if seed control is enabled
                rag_seed = getattr(self.config, 'rag_seed', None) if self.config.enable_seed_control else None

                web_context = make_openrouter_api_call(
                    messages=messages,
                    model=self.config.rag_openrouter_model,
                    api_key=context.config.openrouter_api_key,
                    site_url=context.config.openrouter_site_url,
                    site_name=context.config.openrouter_site_name,
                    temperature=rag_temperature,
                    max_tokens=4000,
                    seed=rag_seed,
                    top_p=rag_top_p,
                    frequency_penalty=rag_frequency_penalty,
                    presence_penalty=rag_presence_penalty
                ).choices[0].message.content.strip().replace('```html', '').replace('```', '').strip()

                logger.debug(f"Generated RAG content: {web_context[:200]}...")

                logger.info(f"Added RAG context to article generation ({len(web_context)} chars)")
                if web_context:
                    context.set_rag_context(web_context)
                logger.info(f"Search results added to RAG context for keyword: {keyword}")
                logger.debug(web_context)

            # Return the initialized context without generating any article components
            return context

        except Exception as e:
            provider.error(f"Error creating article context: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    def generate_article(self, article_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a complete article from the provided data."""
        try:
            keyword = article_data.get('keyword')
            if not keyword:
                provider.error("No keyword provided for article generation")
                return None

            provider.info(f"Generating article for keyword: {keyword}")

            # Create article context with keyword
            context = self.create_article_context(keyword, article_data)
            
            # Ensure context is not None before proceeding
            if context is None:
                provider.error("Failed to create article context")
                return None

            # Set the context in the content generator
            self.content_generator.context = context

            # Initialize web_context as empty string
            web_context = ""

            # Only perform RAG search if both enable_rag and enable_rag_search_engine are True
            if self.config.enable_rag and self.config.enable_rag_search_engine:
                # Perform a search using the keyword
                article_extractor = ArticleExtractor(
                keyword=keyword,
                max_search_results=12,
                max_content_length=2000,
                headless=True
                )

                logger.info("Firing RAG Search")

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
                        "content": f"Gather all the information from the following articles and summarize them in a single place. Make sure to never ever skip any information at any cost.:\n\n{web_context}"
                    }
                ]

                # Get generation parameters from config or use defaults
                rag_temperature = getattr(self.config, 'rag_temperature', 0.7)
                rag_top_p = getattr(self.config, 'rag_top_p', 1.0)
                rag_frequency_penalty = getattr(self.config, 'rag_frequency_penalty', 0.0)
                rag_presence_penalty = getattr(self.config, 'rag_presence_penalty', 0.0)

                # Get seed if seed control is enabled
                rag_seed = getattr(self.config, 'rag_seed', None) if self.config.enable_seed_control else None

                web_context = make_openrouter_api_call(
                    messages=messages,
                    model=self.config.rag_openrouter_model,
                    api_key=context.config.openrouter_api_key,
                    site_url=context.config.openrouter_site_url,
                    site_name=context.config.openrouter_site_name,
                    temperature=rag_temperature,
                    max_tokens=4000,
                    seed=rag_seed,
                    top_p=rag_top_p,
                    frequency_penalty=rag_frequency_penalty,
                    presence_penalty=rag_presence_penalty
                ).choices[0].message.content.strip().replace('```html', '').replace('```', '').strip()

                provider.info(web_context)

                logger.info(f"Added RAG context to article generation ({len(web_context)} chars)")
                if web_context:
                    context.set_rag_context(web_context)
                logger.info(f"Search results added to RAG context for keyword: {keyword}")

            formatting_rules = self.build_formatting_instructions()

            # Generate core components (title, outline, introduction, sections, conclusion)
            article_components = self._generate_core_components(keyword, article_data, context, web_context)
            if not article_components:
                provider.error("Failed to generate core article components")
                return None

            # Generate additional components (PAA, FAQ, meta description, etc.)
            article_components = self._generate_additional_components(keyword, article_components, context, web_context + "\n\n" + formatting_rules)

            # Humanize text if enabled
            if self.config.enable_text_humanization:
                provider.info("Text humanization enabled, humanizing text...")
                article_components = self._humanize_text(article_components, context + "\n\n" + formatting_rules)

            # Check grammar if enabled
            if self.config.enable_grammar_check:
                provider.info("Grammar check enabled, checking grammar...")
                article_components = self._check_grammar(article_components, context)

            # Process images if enabled
            article_components = self._process_images(article_data, article_components)

            if "content" in article_components:
                article_components["content"] = self.clean_html(article_components["content"])
            
            # Save and publish article
            self._save_and_publish_article(keyword, article_components)

            # Save article context if enabled
            if self.config.enable_context_save:
                context.save_to_file()

            # Display token usage statistics if enabled
            if self.config.enable_token_tracking:
                stats = context.get_token_usage_stats()
                provider.display_token_stats(stats)

            provider.success(f"Article generation completed for: {keyword}")
            return article_components

        except Exception as e:
            provider.error(f"Error generating article: {str(e)}")
            provider.error(f"Detailed traceback:\n{traceback.format_exc()}")
            return None

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

    def _generate_core_components(
        self,
        keyword: str,
        article_data: Dict[str, Any],
        context: ArticleContext,
        web_context: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Generate core article components."""
        try:
            provider.info("Generating core article components...")

            # Create a progress bar for core component generation
            with provider.progress_bar() as progress:
                # Build RAG knowledge base once at the beginning if enabled
                if self.config.enable_rag and self.rag_retriever:
                    try:
                        logger.info(f"Building RAG knowledge base for: {keyword}")
                        self.rag_retriever.build_knowledge_base(keyword)
                        # Retrieve relevant content for the keyword
                        relevant_content = self.rag_retriever.retrieve_relevant_content(
                            query=keyword,
                            k=self.config.rag_num_chunks
                        )
                        if relevant_content:
                            # Format RAG context
                            rag_context = "\n\n".join([item["text"] for item in relevant_content])
                            logger.info(f"RAG context built for keyword: {keyword} ({len(rag_context)} characters)")

                            # Add RAG context to the article context system message instead of individual prompts
                            context.set_rag_context(rag_context)

                            # Log the RAG context
                            logger.debug("RAG context:")
                            logger.debug("-" * 80)
                            logger.debug(rag_context)
                            logger.debug("-" * 80)
                        else:
                            logger.warning(f"No relevant content found for keyword: {keyword}")
                    except Exception as e:
                        # Handle RAG failures gracefully - continue without RAG
                        logger.error(f"Error building RAG knowledge base: {str(e)}")
                        logger.warning("Continuing article generation without RAG enhancement")
                        # Disable RAG to prevent further attempts
                        self.config.enable_rag = False
                        self.rag_retriever = None
                elif self.config.enable_rag and not self.rag_retriever:
                    logger.warning("RAG is enabled but retriever is not available. Continuing without RAG.")
                    self.config.enable_rag = False
                formatting_rules = self.build_formatting_instructions()
                # Generate title with a dedicated progress task
                title_task = progress.add_task("[cyan]Generating title...", total=1)
                title = self.content_generator.generate_title(keyword, web_context )
                provider.info(title)
                if not title:
                    provider.error("Failed to generate title")
                    return None
                progress.update(title_task, advance=1)

                # Generate outline with a dedicated progress task
                outline_task = progress.add_task("[cyan]Generating outline...", total=1)
                outline = self.content_generator.generate_outline(keyword, web_context + "\n\n" + formatting_rules)
                if not outline:
                    provider.error("Failed to generate outline")
                    return None
                progress.update(outline_task, advance=1)

                # Generate introduction with a dedicated progress task
                intro_task = progress.add_task("[cyan]Generating introduction...", total=1)
                introduction = self.content_generator.generate_introduction(keyword, title, web_context + "\n\n" + formatting_rules)
                if not introduction:
                    provider.error("Failed to generate introduction")
                    return None
                progress.update(intro_task, advance=1)

            # Generate sections
            sections = []
            headings = []

            # Use subtitles from CSV if available
            # Process all available subtitles from CSV regardless of sizesections
            subtitle_index = 1
            while True:
                subtitle_key = f'subtitle{subtitle_index}'
                if article_data and subtitle_key in article_data and article_data[subtitle_key]:
                    headings.append(article_data[subtitle_key])
                    subtitle_index += 1
                else:
                    break

            # If no subtitles in CSV, use outline to generate them
            if not headings:
                # Extract headings from outline
                outline_lines = outline.strip().split('\n')
                for line in outline_lines:
                    if line.strip() and not line.startswith('#'):
                        headings.append(line.strip())
                        # Don't limit based on sizesections when using outline
                        # Let the outline determine the number of sections

            # Create a progress task for sections
            sections_task = progress.add_task(
                f"[cyan]Generating {len(headings)} sections...", 
                total=len(headings)
            )
            
            # Generate content for each section - no longer passing RAG context
            for i, heading in enumerate(headings, 1):
                section_content = self._generate_sections(keyword, article_data, heading, web_context + "\n\n" + formatting_rules)
                if section_content:
                    sections.append(section_content)
                progress.update(sections_task, advance=1)

            # Generate conclusion with a dedicated progress task
            conclusion_task = progress.add_task("[cyan]Generating conclusion...", total=1)
            conclusion = self.content_generator.generate_conclusion(keyword, title, web_context + "\n\n" + formatting_rules)
            progress.update(conclusion_task, advance=1)
            if not conclusion:
                provider.error("Failed to generate conclusion")
                return None

            # Create article components dictionary
            article_components = {
                'title': title,
                'outline': outline,
                'introduction': introduction,
                'conclusion': conclusion,
                'headings': headings,
                'sections': sections
            }

            # Update the context's article_parts with the generated content
            context.article_parts["title"] = title
            context.article_parts["outline"] = outline
            context.article_parts["introduction"] = introduction
            context.article_parts["conclusion"] = conclusion
            # Add sections to the context
            context.article_parts["sections"] = sections

            provider.success("Core article components generated successfully")
            return article_components

        except Exception as e:
            provider.error(f"Error generating core components: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return None
        
    def _generate_sections(
        self,
        keyword: str,
        article_data: Dict[str, Any],
        heading: str,
        web_context: str = ""
    ) -> Optional[str]:
        """Generate content for a section with multiple paragraphs."""
        try:
            provider.debug(f"Generating section for heading: {heading}")

            # Get section number and total sections from article data
            total_sections = len(article_data.get("headings", []))
            section_number = article_data.get("headings", []).index(heading) + 1 if heading in article_data.get("headings", []) else 1
            
            # Extract section points from outline if available
            section_points = []
            if "outline" in article_data and article_data["outline"]:
                # Try to find section points that match this heading
                for section in article_data["outline"].split('\n'):
                    if heading.lower() in section.lower():
                        # Look for the next section that might contain points
                        section_idx = article_data["outline"].split('\n').index(section)
                        outline_lines = article_data["outline"].split('\n')
                        for i in range(section_idx + 1, len(outline_lines)):
                            if outline_lines[i].strip() and not outline_lines[i].startswith('#'):
                                # This looks like a point
                                point = outline_lines[i].strip()
                                if point.startswith('-'):
                                    point = point[1:].strip()
                                section_points.append(point)
                            # Stop when we hit the next heading
                            if i < len(outline_lines) - 1 and outline_lines[i+1].startswith('#'):
                                break
            
            # If no points were found, create some generic ones based on the heading
            if not section_points:
                section_points = [
                    f"Background information on {heading}",
                    f"Key aspects of {heading}",
                    f"Important details about {heading}",
                    f"Practical applications of {heading}"
                ]

            paragraphs = []
            for i in range(self.config.paragraphs_per_section):
                current_paragraph = i + 1
                
                if self.config.enable_paragraph_headings:
                    # Generate paragraph with heading in a single call
                    formatted_paragraph = self.content_generator.generate_paragraph(
                        keyword, 
                        heading, 
                        current_paragraph=current_paragraph,
                        paragraphs_per_section=self.config.paragraphs_per_section,
                        section_number=section_number,
                        total_sections=total_sections,
                        section_points=section_points,
                        web_context=web_context
                    )
                    if formatted_paragraph:
                        paragraphs.append(formatted_paragraph)
                else:
                    # Use existing paragraph generation without headings
                    paragraph = self.content_generator.generate_paragraph(
                        keyword, 
                        heading, 
                        current_paragraph=current_paragraph,
                        paragraphs_per_section=self.config.paragraphs_per_section,
                        section_number=section_number,
                        total_sections=total_sections,
                        section_points=section_points,
                        web_context=web_context
                    )
                    if paragraph:
                        paragraphs.append(paragraph)

            if not paragraphs:
                context = {
                    "component": "Generator", 
                    "method": "_generate_sections", 
                    "keyword": keyword,
                    "heading": heading
                }
                logger.error(f"No paragraphs generated for heading: {heading}", show_traceback=True)
                return None

            # Combine paragraphs into a single section
            section_content = "\n\n".join(paragraphs)

            # Add the section to the context's messages to maintain conversation history
            self.content_generator.context.add_message("user", f"Generate content for section: {heading}")
            self.content_generator.context.add_message("assistant", section_content)

            return section_content

        except Exception as e:
            context = {
                "component": "Generator", 
                "method": "_generate_sections", 
                "keyword": keyword,
                "heading": heading
            }
            logger.error(f"{e.__class__.__name__}: {str(e)}", show_traceback=True)
            return None

    def _generate_additional_components(
        self,
        keyword: str,
        article_components: Dict[str, Any],
        context: ArticleContext,
        web_context: str = ""
    ) -> Dict[str, Any]:
        """Generate additional article components."""
        try:
            provider.info("Generating additional article components...")
            
            # Create a progress bar for additional components
            with provider.progress_bar() as progress:
                # Generate PAA section if enabled
                if self.config.add_paa_paragraphs_into_article:
                    try:
                        paa_task = progress.add_task("[cyan]Generating PAA section...", total=1)
                        
                        if self.config.enable_rag_search_engine and web_context and web_context.strip() != "":
                            paa_section = self.paa_handler.generate_paa_section(
                                keyword,
                                # serp_api_key=self.config.serp_api_key,
                                article_context=context,
                                web_context=web_context
                            )
                        else:
                            paa_section = self.paa_handler.generate_paa_section(
                                keyword,
                                # serp_api_key=self.config.serp_api_key,
                                article_context=context
                            )

                        if paa_section:
                            article_components['paa_section'] = paa_section
                            # Also add to the context's article_parts
                            context.article_parts["paa"] = paa_section
                            provider.success("PAA section generated successfully")
                        else:
                            error_context = {
                                "component": "Generator", 
                                "method": "_generate_additional_components", 
                                "keyword": keyword,
                                "feature": "PAA"
                            }
                            log_error(Exception("Failed to generate PAA section"), error_context, severity="warning")
                            # If PAA generation fails, disable it to prevent further attempts
                            self.config.add_paa_paragraphs_into_article = False
                            
                        progress.update(paa_task, advance=1)
                    except Exception as e:
                        error_context = {
                            "component": "Generator", 
                            "method": "_generate_additional_components", 
                            "keyword": keyword,
                            "feature": "PAA"
                        }
                        log_error(e, error_context)
                        # Disable PAA on error to prevent further attempts
                        self.config.add_paa_paragraphs_into_article = False
                        provider.warning("PAA feature has been disabled due to errors")
                        progress.update(paa_task, advance=1)

                # Add YouTube video if enabled
                if self.config.add_youtube_video and self.youtube_handler:
                    try:
                        youtube_task = progress.add_task("[cyan]Adding YouTube video...", total=1)
                        
                        # Determine the output format based on WordPress upload setting
                        output_format = 'wordpress' if self.config.enable_wordpress_upload else 'markdown'

                        video_embed = self.youtube_handler.get_video_for_article(keyword, output_format)
                        if video_embed:
                            article_components['youtube_video'] = video_embed
                            provider.success("YouTube video added successfully")
                        else:
                            error_context = {
                                "component": "Generator", 
                                "method": "_generate_additional_components", 
                                "keyword": keyword,
                                "feature": "YouTube"
                            }
                            log_error(Exception("Failed to add YouTube video"), error_context, severity="warning")
                            
                        progress.update(youtube_task, advance=1)
                    except Exception as e:
                        error_context = {
                            "component": "Generator", 
                            "method": "_generate_additional_components", 
                            "keyword": keyword,
                            "feature": "YouTube"
                        }
                        log_error(e, error_context)
                    progress.update(youtube_task, advance=1)
                    
                # Generate FAQ section if enabled
                if self.config.add_faq_into_article:
                    try:
                        faq_task = progress.add_task("[cyan]Generating FAQ section...", total=1)
                        
                        faq_section = self.content_generator.generate_faq(keyword, web_context)
                        if faq_section:
                            article_components['faq_section'] = faq_section
                            # Also add to the context's article_parts
                            context.article_parts["faq"] = faq_section
                            provider.success("FAQ section generated successfully")
                        else:
                            provider.warning("Failed to generate FAQ section")
                            
                        progress.update(faq_task, advance=1)
                    except Exception as e:
                        provider.error(f"Error generating FAQ: {str(e)}")
                        provider.error(f"Stack trace:\n{traceback.format_exc()}")
                        progress.update(faq_task, advance=1)

                # Generate summary if enabled
                if self.config.add_summary_into_article:
                    try:
                        summary_task = progress.add_task("[cyan]Generating article summary...", total=1)
                        
                        # Create article dictionary for summary generation
                        article_dict = {
                            'title': article_components['title'],
                            'introduction': article_components['introduction'],
                            'sections': article_components['sections'],
                            'conclusion': article_components['conclusion']
                        }
                        
                        # Use the dedicated generate_article_summary method from content_generator
                        article_components['summary'] = self.content_generator.generate_article_summary(
                            keyword=keyword,
                            article_dict=article_dict
                        )
                        
                        # Add to the context's article_parts
                        context.article_parts["summary"] = article_components['summary']
                        provider.success(f"Generated article summary ({len(article_components['summary'].split())} words)")
                        
                        progress.update(summary_task, advance=1)

                    except Exception as e:
                        provider.error(f"Error generating article summary: {str(e)}")
                        article_components['summary'] = ""
                        progress.update(summary_task, advance=1)

                # Generate meta description if enabled
                if self.config.enable_meta_description:
                    try:
                        meta_task = progress.add_task("[cyan]Generating meta content...", total=2)
                        
                        meta_description = self.meta_handler.generate_meta_description(keyword, context)
                        if meta_description:
                            article_components['meta_description'] = meta_description
                            # Also add to the context's article_parts
                            context.article_parts["meta_description"] = meta_description
                            provider.success("Meta description generated successfully")
                        else:
                            provider.warning("Failed to generate meta description")
                            
                        progress.update(meta_task, advance=1)
                    except Exception as e:
                        provider.error(f"Error generating meta description: {str(e)}")
                        provider.error(f"Stack trace:\n{traceback.format_exc()}")

                try:
                    wordpress_excerpt = self.meta_handler.generate_wordpress_excerpt(keyword, context)
                    if wordpress_excerpt:
                        article_components['wordpress_excerpt'] = wordpress_excerpt
                        # Also add to the context's article_parts
                        context.article_parts["wordpress_excerpt"] = wordpress_excerpt
                        provider.success("WordPress excerpt generated successfully")
                    else:
                        provider.warning("Failed to generate WordPress excerpt")
                        
                    progress.update(meta_task, advance=1)
                except Exception as e:
                    provider.error(f"Error generating WordPress excerpt: {str(e)}")
                    provider.error(f"Stack trace:\n{traceback.format_exc()}")
                    progress.update(meta_task, advance=1)

                # Generate block notes if enabled
                if self.config.add_blocknote_into_article:
                    try:
                        blocknotes_task = progress.add_task("[cyan]Generating key takeaways...", total=1)
                        
                        # Create article dictionary for block notes generation
                        article_dict_for_notes = {
                            'keyword': keyword,
                            'title': article_components['title'],
                            'introduction': article_components['introduction'],
                            'sections': article_components['sections'],
                            'conclusion': article_components['conclusion']
                        }
                        
                        block_notes = generate_block_notes(
                            context=context,
                            article_content=article_dict_for_notes,
                            blocknote_prompt=self.prompts.blocknote,
                            combine_prompt=self.prompts.blocknotes_combine,
                            engine=self.config.openai_model,
                            enable_token_tracking=self.config.enable_token_tracking,
                            track_token_usage=self.config.enable_token_tracking
                        )
                        
                        if block_notes:
                            # Add to both article_components and context.article_parts
                            article_components['block_notes'] = block_notes
                            context.article_parts['block_notes'] = block_notes
                            provider.success(f"Block notes generated successfully ({len(block_notes)} chars)")
                        else:
                            provider.warning("Failed to generate block notes")
                            
                        progress.update(blocknotes_task, advance=1)
                        
                    except Exception as e:
                        provider.error(f"Error generating block notes: {str(e)}")
                        provider.error(f"Stack trace:\n{traceback.format_exc()}")
                        # Don't silently fail - log the error but continue
                        progress.update(blocknotes_task, advance=1)

                # Generate external links if enabled
                if self.config.add_external_links_into_article:
                    try:
                        links_task = progress.add_task("[cyan]Generating external links...", total=1)
                        
                        from article_generator.external_links_handler import generate_external_links_section

                        # Determine the output format based on WordPress upload setting
                        output_format = 'wordpress' if self.config.enable_wordpress_upload else 'markdown'

                        external_links = generate_external_links_section(
                            keyword=keyword,
                            config=self.config,
                            output_format=output_format
                        )

                        if external_links:
                            article_components['external_links'] = external_links
                            provider.success("External links section generated successfully")
                        else:
                            provider.warning("Failed to generate external links section")
                            
                        progress.update(links_task, advance=1)
                    except Exception as e:
                        provider.error(f"Error generating external links: {str(e)}")
                        provider.error(f"Stack trace:\n{traceback.format_exc()}")
                        progress.update(links_task, advance=1)

            provider.success("Additional components generated successfully")
            return article_components

        except Exception as e:
            provider.error(f"Error generating additional components: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return article_components

    def _process_images(
        self,
        article_data: Dict[str, Any],
        article_components: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process images for the article."""
        try:
            keyword = article_data.get('keyword', '')

            # Skip image processing if disabled
            if not self.config.enable_image_generation:
                provider.info("Image generation is disabled, skipping...")
                return article_components

            provider.info(f"Processing images for keyword: {keyword}")

            # Create a progress bar for image processing
            with provider.progress_bar() as progress:
                feature_image_task = progress.add_task("[cyan]Generating feature image...", total=1)
                body_images_task = progress.add_task("[cyan]Generating body images...", total=1)
                
                # Create timestamp for unique image filenames
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

                # Create ImageConfig from the flat config with all image parameters
                image_config = ImageConfig(
                    enable_image_generation=self.config.enable_image_generation,
                    max_number_of_images=self.config.max_number_of_images,
                    orientation=self.config.orientation,
                    order_by=self.config.order_by,
                    
                    image_source=self.config.image_source,
                    stock_primary_source=self.config.stock_primary_source,
                    secondary_source_image=self.config.secondary_source_image,
                    image_api=self.config.image_api,
                    
                    # AI Image settings
                    huggingface_model=self.config.huggingface_model,
                    huggingface_api_key=self.config.huggingface_api_key,
                    
                    # Image caption settings
                    image_caption_instance=self.config.image_caption_instance,
                    
                    
                    # Image sources api keys
                    unsplash_api_key=self.config.unsplash_api_key,
                    pexels_api_key=self.config.pexels_api_key,
                    pixabay_api_key=self.config.pixabay_api_key,
                    giphy_api_key=self.config.giphy_api_key,
                   
                    # Add the new image parameters
                    alignment=self.config.image_alignment,
                    enable_image_compression=self.config.enable_image_compression,
                    compression_quality=self.config.image_compression_quality,
                    prevent_duplicate_images=self.config.prevent_duplicate_images
                )

            feature_image = None
            body_images = []
            used_images = []

            # Process feature image from CSV
            if 'featured_img' in article_data and article_data['featured_img']:
                try:
                    featured_img_keyword = article_data['featured_img']
                    provider.info(f"Processing feature image from CSV: {featured_img_keyword}")
                    if image_config.image_source == "imageai":
                        images,sourceUsed = fetch_image(image_config,image_config.image_source,image_config.stock_primary_source,featured_img_keyword,2,image_config.secondary_source_image)
                    else:
                        images,sourceUsed = fetch_image(image_config,image_config.image_source,image_config.stock_primary_source,featured_img_keyword,image_config.max_number_of_images,image_config.secondary_source_image)

                    # Get image from any using the keyword
                    indexOfImage = image_randomizer(images)
                    # images = get_image_list_by_source(featured_img_keyword, image_config, 1)
                    if images:
                        feature_image = process_feature_image(images[indexOfImage], keyword, sourceUsed, None, timestamp)
                        if feature_image:
                            provider.success("Feature image processed successfully")
                            used_images.append(images[indexOfImage])
                        else:
                            provider.warning("Failed to process feature image from CSV")
                    else:
                        provider.warning(f"No images found for feature image keyword: {featured_img_keyword}")
                except Exception as img_error:
                    provider.error(f"Error processing feature image from CSV: {str(img_error)}")
                    provider.error(f"Detailed traceback:\n{traceback.format_exc()}")
            
            # Update feature image task progress
            progress.update(feature_image_task, advance=1)

          
            # Process body images from CSV
            for i in range(0, image_config.max_number_of_images):  # Process up to 5 body images
                img_key = f'img{i}'
                if img_key in article_data and article_data[img_key]:
                    try:
                        img_keyword = article_data[img_key]
                        provider.info(f"Processing body image {i} from CSV: {img_keyword}")

                        # Get image from any source using the keyword
                        # images = get_image_list_by_source(img_keyword, image_config, 1)
                        if image_config.image_source == "imageai":
                           images,sourceUsed = fetch_image(image_config,image_config.image_source,image_config.stock_primary_source,img_keyword,2,image_config.secondary_source_image)
                        else:
                           images,sourceUsed = fetch_image(image_config,image_config.image_source,image_config.stock_primary_source,img_keyword,image_config.max_number_of_images,image_config.secondary_source_image)

                        
                        count = 0
                        images_selected = []
                        for image in range(0, len(images)):
                            randomizedIndex = image_randomizer(images)
                            if images[randomizedIndex] not in used_images:
                                rand_value = randomizedIndex
                                used_images.append(images[rand_value])
                                images_selected.append(images[rand_value])
                                count+=1
                                if count == 4:
                                   break
                           
                        if images_selected:
                            for each in range(0, len(images_selected)):
                                body_image = process_body_image(images_selected[each], keyword, sourceUsed, i, save_dir=None, timestamp=timestamp)
                                if body_image:
                                    body_images.append(body_image)
                                    provider.success(f"Body image {len(body_images)} processed successfully")
                                else:
                                    provider.warning(f"Failed to process body image {len(body_images)} from CSV")
                        else:
                            provider.warning(f"No images found for body image keyword: {img_keyword}")
                    except Exception as img_error:
                        provider.error(f"Error processing body image {i} from CSV: {str(img_error)}")
                        provider.error(f"Detailed traceback:\n{traceback.format_exc()}")

            # Add images to article components
            article_components['feature_image'] = feature_image
            article_components['body_images'] = body_images
            
            # Update body images task progress
            progress.update(body_images_task, advance=1)

            provider.info(f"Processed {len(body_images)} body images")
            return article_components

        except Exception as e:
            provider.error(f"Error processing images: {str(e)}")
            provider.error(f"Detailed traceback:\n{traceback.format_exc()}")
            # Return original components without images
            return article_components

    def _check_grammar(
        self,
        article_components: Dict[str, Any],
        context: ArticleContext
    ) -> Dict[str, Any]:
        """
        Check grammar for all article components.

        Args:
            article_components (Dict[str, Any]): Article components
            context (ArticleContext): Article context

        Returns:
            Dict[str, Any]: Updated article components with grammar checked
        """
        try:
            provider.info("Checking grammar for article components...")

            from article_generator.text_processor import check_grammar
            
            # Create a progress bar for grammar checking
            with provider.progress_bar() as progress:
                # Create individual tasks for each component
                title_task = progress.add_task("[cyan]Checking title grammar...", total=1)
                intro_task = progress.add_task("[cyan]Checking introduction grammar...", total=1)
                sections_task = progress.add_task("[cyan]Checking sections grammar...", total=1)
                conclusion_task = progress.add_task("[cyan]Checking conclusion grammar...", total=1)
                faq_task = progress.add_task("[cyan]Checking FAQ grammar...", total=1)
                paa_task = progress.add_task("[cyan]Checking PAA grammar...", total=1)
                
                # Check title grammar
                if 'title' in article_components and article_components['title']:
                    article_components['title'] = check_grammar(
                        context,
                        article_components['title'],
                        self.prompts.grammar,
                        engine=self.config.openai_model,
                        enable_token_tracking=self.config.enable_token_tracking,
                        track_token_usage=self.config.enable_token_tracking,
                        content_type="Title"
                    )
                    progress.update(title_task, advance=1)

                # Check introduction grammar
                if 'introduction' in article_components and article_components['introduction']:
                    article_components['introduction'] = check_grammar(
                        context,
                        article_components['introduction'],
                        self.prompts.grammar,
                        engine=self.config.openai_model,
                        enable_token_tracking=self.config.enable_token_tracking,
                        track_token_usage=self.config.enable_token_tracking,
                        content_type="Introduction"
                    )
                    progress.update(intro_task, advance=1)

                # Check sections grammar paragraph by paragraph
                if 'sections' in article_components and article_components['sections']:
                    for i, section in enumerate(article_components['sections']):
                        if section:
                            # Split the section into paragraphs
                            paragraphs = section.split('\n\n')
                            processed_paragraphs = []
                            
                            for paragraph in paragraphs:
                                paragraph = paragraph.strip()
                                if not paragraph:
                                    processed_paragraphs.append("")
                                    continue
                                
                                # Skip headings (lines starting with #, ##, ###)
                                if paragraph.startswith('#'):
                                    processed_paragraphs.append(paragraph)
                                    continue
                                
                                # Process regular paragraph content
                                processed_paragraph = check_grammar(
                                    context,
                                    paragraph,
                                    self.prompts.grammar,
                                    engine=self.config.openai_model,
                                    enable_token_tracking=self.config.enable_token_tracking,
                                    track_token_usage=self.config.enable_token_tracking,
                                    content_type=f"Section {i+1} paragraph"
                                )
                                processed_paragraphs.append(processed_paragraph)
                            
                            # Recombine the paragraphs
                            article_components['sections'][i] = '\n\n'.join(processed_paragraphs)
                    
                    progress.update(sections_task, advance=1)

                # Check conclusion grammar
                if 'conclusion' in article_components and article_components['conclusion']:
                    article_components['conclusion'] = check_grammar(
                        context,
                        article_components['conclusion'],
                        self.prompts.grammar,
                        engine=self.config.openai_model,
                        enable_token_tracking=self.config.enable_token_tracking,
                        track_token_usage=self.config.enable_token_tracking,
                        content_type="Conclusion"
                    )
                    progress.update(conclusion_task, advance=1)

                # Check FAQ section grammar if present - preserve WordPress block structure
                if 'faq_section' in article_components and article_components['faq_section']:
                    # Parse FAQ section to preserve WordPress block structure
                    faq_content = article_components['faq_section']
                    faq_lines = faq_content.split('\n')
                    grammar_checked_lines = []
                    
                    for line in faq_lines:
                        line = line.strip()
                        if not line:
                            grammar_checked_lines.append('')
                            continue
                            
                        # Preserve WordPress blocks and HTML structure
                        if (line.startswith('<!-- wp:') or 
                            line.startswith('<h') or 
                            line.startswith('</h') or 
                            line.startswith('<!-- /wp:')):
                            grammar_checked_lines.append(line)
                        elif line.startswith('<p>') and line.endswith('</p>'):
                            # Extract content from paragraph tags for grammar checking
                            content = line[3:-4]  # Remove <p> and </p>
                            if content.strip():
                                try:
                                    grammar_checked_content = check_grammar(
                                        context,
                                        content,
                                        self.prompts.grammar,
                                        engine=self.config.openai_model,
                                        enable_token_tracking=False,  # Avoid token spam
                                        track_token_usage=False,
                                        content_type="FAQ Answer"
                                    )
                                    grammar_checked_lines.append(f"<p>{grammar_checked_content}</p>")
                                except Exception as e:
                                    provider.warning(f"Error grammar checking FAQ answer: {str(e)}")
                                    grammar_checked_lines.append(line)  # Keep original on error
                            else:
                                grammar_checked_lines.append(line)
                        else:
                            # For any other content, preserve as-is
                            grammar_checked_lines.append(line)
                    
                    article_components['faq_section'] = '\n'.join(grammar_checked_lines)
                    progress.update(faq_task, advance=1)

                # Check PAA section grammar if present - preserve markdown structure
                if 'paa_section' in article_components and article_components['paa_section']:
                    # Parse PAA section to preserve markdown structure
                    paa_content = article_components['paa_section']
                    paa_paragraphs = []
                    current_paragraph = []
                    is_header = False
                    
                    # Split by lines
                    paa_lines = paa_content.split('\n')
                    
                    i = 0
                    while i < len(paa_lines):
                        line = paa_lines[i].rstrip()
                        
                        # Handle headings - preserve as-is
                        if line.startswith('#'):
                            # If we were building a paragraph, finalize it before starting new section
                            if current_paragraph:
                                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                                current_paragraph = []
                            
                            current_paragraph.append(line)
                            is_header = True
                            
                            # Add blank line after header if present
                            if i + 1 < len(paa_lines) and not paa_lines[i + 1].strip():
                                current_paragraph.append('')
                                i += 1
                        
                        # Empty line - potential paragraph separator
                        elif not line:
                            if current_paragraph:
                                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                                current_paragraph = []
                                is_header = False
                            current_paragraph.append('')
                        
                        # Regular content line
                        else:
                            # If we were in a header and now we're not, finalize header paragraph
                            if is_header and current_paragraph:
                                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                                current_paragraph = []
                                is_header = False
                            
                            # Add this content line to current paragraph
                            current_paragraph.append(line)
                        
                        i += 1
                    
                    # Don't forget the last paragraph if there is one
                    if current_paragraph:
                        paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                    
                    # Process each paragraph
                    processed_paragraphs = []
                    for paragraph_text, is_heading in paa_paragraphs:
                        if is_heading or not paragraph_text.strip():
                            # Skip processing for headings and empty lines
                            processed_paragraphs.append(paragraph_text)
                        else:
                            # Process paragraph content
                            try:
                                processed_paragraph = check_grammar(
                                    context,
                                    paragraph_text,
                                    self.prompts.grammar,
                                    engine=self.config.openai_model,
                                    enable_token_tracking=False,  # Avoid token spam
                                    track_token_usage=False,
                                    content_type="PAA Answer Paragraph"
                                )
                                processed_paragraphs.append(processed_paragraph)
                            except Exception as e:
                                provider.warning(f"Error grammar checking PAA paragraph: {str(e)}")
                                processed_paragraphs.append(paragraph_text)  # Keep original on error
                    
                    # Rebuild the PAA section with processed paragraphs - using double newlines for proper paragraph separation
                    rebuilt_content = []
                    for i, paragraph in enumerate(processed_paragraphs):
                        rebuilt_content.append(paragraph)
                        # Add double newline between paragraphs, but not after empty lines or headers
                        if (i < len(processed_paragraphs) - 1 and 
                            paragraph.strip() and not paragraph.strip().startswith('#') and
                            not (i+1 < len(processed_paragraphs) and not processed_paragraphs[i+1].strip())):
                            rebuilt_content.append('')  # This creates a blank line between paragraphs
                    
                    article_components['paa_section'] = '\n'.join(rebuilt_content)
                    progress.update(paa_task, advance=1)

                # Create tasks for summary and block notes if needed
                summary_task = progress.add_task("[cyan]Checking summary grammar...", total=1)
                blocknotes_task = progress.add_task("[cyan]Checking block notes grammar...", total=1)
                
                # Skip summary if present - preserve as is
                if 'summary' in article_components and article_components['summary']:
                    progress.update(summary_task, advance=1)

                # Skip block notes if present - preserve as is
                if 'block_notes' in article_components and article_components['block_notes']:
                    progress.update(blocknotes_task, advance=1)

                provider.success("Grammar check completed for all article components")
                return article_components

        except Exception as e:
            provider.error(f"Error in grammar checking: {str(e)}")
            provider.error(f"Detailed traceback:\n{traceback.format_exc()}")
            return article_components

    def _humanize_text(
        self,
        article_components: Dict[str, Any],
        context: ArticleContext
    ) -> Dict[str, Any]:
        """
        Humanize text for all article components.

        Args:
            article_components (Dict[str, Any]): Article components
            context (ArticleContext): Article context

        Returns:
            Dict[str, Any]: Updated article components with humanized text
        """
        try:
            provider.info("Humanizing text for article components...")

            from article_generator.text_processor import humanize_text
            
            # Create a progress bar for humanization
            with provider.progress_bar() as progress:
                # Create individual tasks for each component
                title_task = progress.add_task("[cyan]Humanizing title...", total=1)
                intro_task = progress.add_task("[cyan]Humanizing introduction...", total=1)
                sections_task = progress.add_task("[cyan]Humanizing sections...", total=1)
                conclusion_task = progress.add_task("[cyan]Humanizing conclusion...", total=1)
                faq_task = progress.add_task("[cyan]Humanizing FAQ...", total=1)
                paa_task = progress.add_task("[cyan]Humanizing PAA...", total=1)

                # Humanize title
                if 'title' in article_components and article_components['title']:
                    article_components['title'] = humanize_text(
                        context,
                        article_components['title'],
                        self.prompts.humanize,
                        engine=self.config.openai_model,
                        enable_token_tracking=self.config.enable_token_tracking,
                        track_token_usage=self.config.enable_token_tracking,
                        content_type="Title"
                    )
                    progress.update(title_task, advance=1)

                # Humanize introduction
                if 'introduction' in article_components and article_components['introduction']:
                    article_components['introduction'] = humanize_text(
                        context,
                        article_components['introduction'],
                        self.prompts.humanize,
                        engine=self.config.openai_model,
                        enable_token_tracking=self.config.enable_token_tracking,
                        track_token_usage=self.config.enable_token_tracking,
                        content_type="Introduction"
                    )
                    progress.update(intro_task, advance=1)

                # Humanize sections paragraph by paragraph
                if 'sections' in article_components and article_components['sections']:
                    for i, section in enumerate(article_components['sections']):
                        if section:
                            # Split the section into paragraphs
                            paragraphs = section.split('\n\n')
                            processed_paragraphs = []
                            
                            for paragraph in paragraphs:
                                paragraph = paragraph.strip()
                                if not paragraph:
                                    processed_paragraphs.append("")
                                    continue
                                
                                # Skip headings (lines starting with #, ##, ###)
                                if paragraph.startswith('#'):
                                    processed_paragraphs.append(paragraph)
                                    continue
                                
                                # Process regular paragraph content
                                processed_paragraph = humanize_text(
                                    context,
                                    paragraph,
                                    self.prompts.humanize,
                                    engine=self.config.openai_model,
                                    enable_token_tracking=self.config.enable_token_tracking,
                                    track_token_usage=self.config.enable_token_tracking,
                                    content_type=f"Section {i+1} paragraph"
                                )
                                processed_paragraphs.append(processed_paragraph)
                            
                            # Recombine the paragraphs
                            article_components['sections'][i] = '\n\n'.join(processed_paragraphs)
                    
                    progress.update(sections_task, advance=1)

                # Humanize conclusion
                if 'conclusion' in article_components and article_components['conclusion']:
                    article_components['conclusion'] = humanize_text(
                        context,
                        article_components['conclusion'],
                        self.prompts.humanize,
                        engine=self.config.openai_model,
                        enable_token_tracking=self.config.enable_token_tracking,
                        track_token_usage=self.config.enable_token_tracking,
                        content_type="Conclusion"
                    )
                    progress.update(conclusion_task, advance=1)

                # Humanize FAQ section if present - preserve WordPress block structure
                if 'faq_section' in article_components and article_components['faq_section']:
                    # Parse FAQ section to preserve WordPress block structure
                    faq_content = article_components['faq_section']
                    faq_lines = faq_content.split('\n')
                    humanized_lines = []
                    
                    for line in faq_lines:
                        line = line.strip()
                        if not line:
                            humanized_lines.append('')
                            continue
                            
                        # Preserve WordPress blocks and HTML structure
                        if (line.startswith('<!-- wp:') or 
                            line.startswith('<h') or 
                            line.startswith('</h') or 
                            line.startswith('<!-- /wp:')):
                            humanized_lines.append(line)
                        elif line.startswith('<p>') and line.endswith('</p>'):
                            # Extract content from paragraph tags for humanization
                            content = line[3:-4]  # Remove <p> and </p>
                            if content.strip():
                                try:
                                    humanized_content = humanize_text(
                                        context,
                                        content,
                                        self.prompts.humanize,
                                        engine=self.config.openai_model,
                                        enable_token_tracking=False,  # Avoid token spam
                                        track_token_usage=False,
                                        content_type="FAQ Answer"
                                    )
                                    humanized_lines.append(f"<p>{humanized_content}</p>")
                                except Exception as e:
                                    provider.warning(f"Error humanizing FAQ answer: {str(e)}")
                                    humanized_lines.append(line)  # Keep original on error
                            else:
                                humanized_lines.append(line)
                        else:
                            # For any other content, preserve as-is
                            humanized_lines.append(line)
                    
                    article_components['faq_section'] = '\n'.join(humanized_lines)
                    progress.update(faq_task, advance=1)

                # Humanize PAA section if present - preserve markdown structure
                if 'paa_section' in article_components and article_components['paa_section']:
                    # Parse PAA section to preserve structure
                    paa_content = article_components['paa_section']
                    paa_paragraphs = []
                    current_paragraph = []
                    is_header = False
                    
                    # Split by lines
                    paa_lines = paa_content.split('\n')
                    
                    i = 0
                    while i < len(paa_lines):
                        line = paa_lines[i].rstrip()
                        
                        # Handle headings - preserve as-is
                        if line.startswith('#'):
                            # If we were building a paragraph, finalize it before starting new section
                            if current_paragraph:
                                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                                current_paragraph = []
                            
                            current_paragraph.append(line)
                            is_header = True
                            
                            # Add blank line after header if present
                            if i + 1 < len(paa_lines) and not paa_lines[i + 1].strip():
                                current_paragraph.append('')
                                i += 1
                        
                        # Empty line - potential paragraph separator
                        elif not line:
                            if current_paragraph:
                                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                                current_paragraph = []
                                is_header = False
                            current_paragraph.append('')
                        
                        # Regular content line
                        else:
                            # If we were in a header and now we're not, finalize header paragraph
                            if is_header and current_paragraph:
                                paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                                current_paragraph = []
                                is_header = False
                            
                            # Add this content line to current paragraph
                            current_paragraph.append(line)
                        
                        i += 1
                    
                    # Don't forget the last paragraph if there is one
                    if current_paragraph:
                        paa_paragraphs.append(('\n'.join(current_paragraph), is_header))
                    
                    # Process each paragraph
                    processed_paragraphs = []
                    for paragraph_text, is_heading in paa_paragraphs:
                        if is_heading or not paragraph_text.strip():
                            # Skip processing for headings and empty lines
                            processed_paragraphs.append(paragraph_text)
                        else:
                            # Process paragraph content
                            try:
                                processed_paragraph = humanize_text(
                                    context,
                                    paragraph_text,
                                    self.prompts.humanize,
                                    engine=self.config.openai_model,
                                    enable_token_tracking=False,  # Avoid token spam
                                    track_token_usage=False,
                                    content_type="PAA Answer Paragraph"
                                )
                                processed_paragraphs.append(processed_paragraph)
                            except Exception as e:
                                provider.warning(f"Error humanizing PAA paragraph: {str(e)}")
                                processed_paragraphs.append(paragraph_text)  # Keep original on error
                    
                    # Rebuild the PAA section with processed paragraphs - using double newlines for proper paragraph separation
                    rebuilt_content = []
                    for i, paragraph in enumerate(processed_paragraphs):
                        rebuilt_content.append(paragraph)
                        # Add double newline between paragraphs, but not after empty lines or headers
                        if (i < len(processed_paragraphs) - 1 and 
                            paragraph.strip() and not paragraph.strip().startswith('#') and
                            not (i+1 < len(processed_paragraphs) and not processed_paragraphs[i+1].strip())):
                            rebuilt_content.append('')  # This creates a blank line between paragraphs
                    
                    article_components['paa_section'] = '\n'.join(rebuilt_content)
                    progress.update(paa_task, advance=1)

                # Create tasks for summary and block notes if needed
                summary_task = progress.add_task("[cyan]Checking summary...", total=1)
                blocknotes_task = progress.add_task("[cyan]Checking block notes...", total=1)

                # Skip block notes if present - preserve as is
                if 'block_notes' in article_components and article_components['block_notes']:
                    progress.update(blocknotes_task, advance=1)

                # Skip summary if present - preserve as is
                if 'summary' in article_components and article_components['summary']:
                    progress.update(summary_task, advance=1)

                provider.success("Text humanization completed for all article components")
                return article_components

        except Exception as e:
            provider.error(f"Error in text humanization: {str(e)}")
            provider.error(f"Detailed traceback:\n{traceback.format_exc()}")
            return article_components

    def _save_and_publish_article(
        self,
        keyword: str,
        article_components: Dict[str, Any]
    ) -> None:
        """Save the article as markdown and publish to WordPress if enabled."""
        try:
            # Save as markdown
            if self.config.enable_markdown_save:
                markdown_path = self.save_as_markdown(article_components, keyword)
                provider.info(f"Article saved as markdown: {markdown_path}")

            # Publish to WordPress if enabled
            if self.config.enable_wordpress_upload and article_components:
                provider.info("Publishing to WordPress...")

                # Publish to WordPress without tags
                self.publish_to_wordpress(article_components)

        except Exception as e:
            provider.error(f"Error saving/publishing article: {str(e)}")
            provider.error(f"Detailed traceback:\n{traceback.format_exc()}")

    def save_as_markdown(self, article_dict: dict, keyword: str) -> str:
        """
        Save article as markdown file.

        Args:
            article_dict (dict): Article components
            keyword (str): Article keyword

        Returns:
            str: Path to saved markdown file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.config.markdown_output_dir, exist_ok=True)

            # Create filename from keyword
            filename = keyword.lower().replace(' ', '_').replace('-', '_')
            filepath = os.path.join(self.config.markdown_output_dir, f"{filename}.md")

            provider.info(f"Saving article as markdown: {filepath}")

            # Build markdown content
            markdown_content = f"# {article_dict['title']}\n\n"

            # Add feature image if available
            if 'feature_image' in article_dict and article_dict['feature_image']:
                # Use the original URL from Unsplash for the markdown file
                img_url = article_dict['feature_image'].get('url', '')
                img_alt = article_dict['feature_image'].get('alt', keyword)
                img_caption = article_dict['feature_image'].get('caption', '')

                markdown_content += f"![{img_alt}]({img_url})\n"
                if img_caption:
                    markdown_content += f"*{img_caption}*\n\n"
                else:
                    markdown_content += "\n\n"

            # Add introduction
            if 'introduction' in article_dict and article_dict['introduction']:
                markdown_content += f"{article_dict['introduction']}\n\n"

            # Add YouTube video after introduction if configured
            if self.config.youtube_position == "after_introduction" and 'youtube_video' in article_dict and article_dict['youtube_video']:
                markdown_content += f"{article_dict['youtube_video']}\n\n"

            # Add sections with headings
            if 'headings' in article_dict and 'sections' in article_dict:
                for i, (heading, section) in enumerate(zip(article_dict['headings'], article_dict['sections'])):
                    if heading and section:
                        markdown_content += f"## {heading}\n\n"

                        # Add body image at the beginning of the section if available
                        if 'body_images' in article_dict and article_dict['body_images'] and i < len(article_dict['body_images']):
                            img = article_dict['body_images'][i]
                            img_url = img.get('url', '')
                            img_alt = img.get('alt', f"Image {i+1}")
                            img_caption = img.get('caption', '')

                            markdown_content += f"![{img_alt}]({img_url})\n"
                            if img_caption:
                                markdown_content += f"*{img_caption}*\n\n"
                            else:
                                markdown_content += "\n\n"

                        # Add YouTube video after first section if configured
                        if i == 0 and self.config.youtube_position == "after_first_section" and 'youtube_video' in article_dict and article_dict['youtube_video']:
                            markdown_content += f"{article_dict['youtube_video']}\n\n"

                        # Add section content
                        markdown_content += f"{section}\n\n"

            # Add conclusion
            if 'conclusion' in article_dict and article_dict['conclusion']:
                markdown_content += f"## Conclusion\n\n{article_dict['conclusion']}\n\n"

            # Add YouTube video at end if configured
            if self.config.youtube_position == "end" and 'youtube_video' in article_dict and article_dict['youtube_video']:
                markdown_content += f"{article_dict['youtube_video']}\n\n"

            # Add PAA section if available
            if 'paa_section' in article_dict and article_dict['paa_section']:
                # Remove any existing heading to avoid duplication
                paa_content = article_dict['paa_section']
                paa_content = re.sub(r'^## People Also Ask\s*', '', paa_content, flags=re.MULTILINE)

                markdown_content += f"## People Also Ask\n\n{paa_content}\n\n"

            # Add FAQ section if available
            if 'faq_section' in article_dict and article_dict['faq_section']:
                # Remove any existing heading to avoid duplication
                faq_content = article_dict['faq_section']
                faq_content = re.sub(r'^## Frequently Asked Questions\s*', '', faq_content, flags=re.MULTILINE)

                markdown_content += f"## Frequently Asked Questions\n\n{faq_content}\n\n"

            # Add external links if available
            if 'external_links' in article_dict and article_dict['external_links']:
                # For markdown, the external_links_handler already formats it correctly
                markdown_content += f"{article_dict['external_links']}\n\n"

            # Add block notes (Key Takeaways) if available
            if 'block_notes' in article_dict and article_dict['block_notes']:
                # Remove any existing heading to avoid duplication
                block_notes_content = article_dict['block_notes']
                block_notes_content = re.sub(r'^## Key Takeaways\s*', '', block_notes_content, flags=re.MULTILINE)

                markdown_content += f"## Key Takeaways\n\n{block_notes_content}\n\n"

            # Add summary if available
            if 'summary' in article_dict and article_dict['summary']:
                markdown_content += f"## Summary\n\n{article_dict['summary']}\n\n"

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            provider.success(f"Article saved as markdown: {filepath}")
            return filepath

        except Exception as e:
            provider.error(f"Error saving markdown: {str(e)}")
            provider.error(f"Detailed traceback:\n{traceback.format_exc()}")
            return ""

    def publish_to_wordpress(self, article_components: Dict[str, str], tags: Optional[List[str]] = None) -> None:
        """
        Publish article to WordPress.

        Args:
            article_components (Dict[str, str]): Article components
            tags (Optional[List[str]]): List of tags
        """
        try:
            if not self.config.enable_wordpress_upload:
                provider.info("WordPress upload is disabled")
                return

            provider.info("Publishing article to WordPress...")

            # Get article data
            keyword = article_components.get('keyword', '')
            title = article_components.get('title', '')

            # Check for body images
            body_images = []
            if 'body_images' in article_components and article_components['body_images']:
                provider.info(f"Found {len(article_components['body_images'])} body images")
                body_images = article_components['body_images']

            # Check for summary
            if 'summary' in article_components and article_components['summary']:
                provider.info("Adding summary to article content")

            # Format article for WordPress
            content = format_article_for_wordpress(
                self.config,
                article_components,
                body_images=body_images,
                add_summary=self.config.add_summary_into_article,
                add_block_notes=self.config.add_blocknote_into_article
            )

            # Prepare article data
            article_data = {
                'title': title,
                'content': content,
                'keyword': keyword
            }

            # Get feature image
            feature_image = None
            if 'feature_image' in article_components and article_components['feature_image']:
                feature_image = article_components['feature_image'].get('file')

            # Get meta description
            meta_description = None
            if 'meta_description' in article_components and article_components['meta_description']:
                meta_description = article_components['meta_description']

            # Get WordPress excerpt
            wordpress_excerpt = None
            if 'wordpress_excerpt' in article_components and article_components['wordpress_excerpt']:
                wordpress_excerpt = article_components['wordpress_excerpt']
            

            # Post to WordPress
            result = post_to_wordpress(
                self.image_config,
                website_name=self.config.WP_WEBSITE_NAME,
                Username=self.config.WP_USERNAME,
                App_pass=self.config.wp_app_pass,
                categories=self.config.wp_categories,
                author=self.config.wp_custom_author if self.config.wp_custom_author else self.config.wp_author,
                status=self.config.wp_post_status,
                article=article_data,
                feature_image=feature_image,
                body_images=body_images,
                meta_description=meta_description,
                wordpress_excerpt=wordpress_excerpt,
                tags=tags,
                keyword=keyword
            )

            if result:
                provider.success("Article published to WordPress successfully")
            else:
                provider.error("Failed to publish article to WordPress")

        except Exception as e:
            provider.error(f"Error publishing to WordPress: {str(e)}")
            provider.error(f"Detailed traceback:\n{traceback.format_exc()}")