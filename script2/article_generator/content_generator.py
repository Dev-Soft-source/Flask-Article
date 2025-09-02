# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from typing import Dict, Any, Optional, List
import re
import json
import traceback
from openai import OpenAI
from config import Config
from utils.api_utils import RetryHandler
from utils.text_utils import TextProcessor
from utils.prompts_config import Prompts
from utils.rich_provider import provider
from utils.rate_limiter import openai_rate_limiter, initialize_rate_limiters, RateLimitConfig
from utils.ai_utils import generate_completion, make_openrouter_api_call
from .article_context import ArticleContext
from .rag_retriever import WebContentRetriever
from .logger import logger
from .chunking_utils import chunk_article_for_processing, combine_chunk_results, combine_chunk_results_with_llm
from bs4 import BeautifulSoup


class ContentGenerator:
    """Handles AI-powered content generation for articles."""

    def __init__(self, config: Config, prompts: Prompts):
        """Initialize the content generator with configuration."""
        self.config = config
        self.prompts = prompts
        self.client = OpenAI(api_key=config.openai_key)
        self.retry_handler = RetryHandler(config)
        self.text_processor = TextProcessor(config)
        # We'll use the context passed from the Generator instead of creating our own
        self.context = None

        # Initialize rate limiters if not already done
        if hasattr(self.config, 'enable_rate_limiting') and self.config.enable_rate_limiting:
            # Check if rate limiters need initialization
            if openai_rate_limiter is None:
                logger.info("Initializing rate limiters in ContentGenerator")
                initialize_rate_limiters(
                    openai_config=RateLimitConfig(
                        rpm=getattr(self.config, 'openai_rpm', 60),
                        rpd=getattr(self.config, 'openai_rpd', 10000),
                        enabled=True
                    )
                )

        # Log the API provider being used
        if config.use_openrouter and config.openrouter_api_key:
            logger.info(f"Using OpenRouter for API calls with site: {config.openrouter_site_name}")
        else:
            logger.info(f"Using OpenAI direct API with model: {config.openai_model}")

        # Initialize RAG if enabled
        if config.enable_rag:
            logger.info("Initializing RAG system...")
            self.rag_retriever = WebContentRetriever(config)
            logger.success("RAG system initialized successfully")
        else:
            self.rag_retriever = None
            logger.info("RAG system disabled")

        provider.debug("ContentGenerator initialized successfully")

    def generate_title(self, keyword: str, web_context:str = "") -> str:
        """Generates an engaging title for the article."""
        try:
            provider.info(f"Generating title for keyword: {keyword}")

            # Prepare prompt without explicit RAG context
            prompt = self.prompts.format_prompt(
                'title',
                keyword=keyword,
                articlelanguage=self.config.articlelanguage,
                voicetone=self.config.voicetone,
                articletype=self.config.articletype,
                articleaudience=self.config.articleaudience,
                pointofview=self.config.pointofview,
                sizeheadings=self.config.sizeheadings,
                context_summary=self.context.get_context_summary() if self.context else "",
            )

            if web_context != "":
                prompt += f"\n\nFollow this Web context and use the data in it: {web_context}"

            # Use title seed if seed control is enabled
            seed = self.config.title_seed if self.config.enable_seed_control else None

            self.context.add_message("user", prompt)
            response = self._generate_content(
                prompt,
                max_tokens=self.config.token_limits['title'],
                generation_type="content_generation",
                seed=seed
            )

            # Clean up any markdown artifacts and extra whitespace
            title = response.replace('*', '').replace('#', '').replace('"', '').strip()

            # Display generated title
            provider.success(f"Generated title: {title}")

            # Display token usage if enabled
            if self.config.enable_token_tracking:
                provider.token_usage(f"Title generation - Tokens used: {self.context.count_message_tokens({'role': 'assistant', 'content': title})}")

            # Don't add to context again - generate_completion already did that
            # Just update the article parts
            self.context.article_parts["title"] = title
            return title
        except Exception as e:
            provider.error(f"Error generating title: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return ""

    def generate_outline(self, keyword: str, web_context: str = "") -> str:
        """Generates an article outline based on the keyword."""
        try:
            provider.info(f"Generating outline for keyword: {keyword}")

            # Import the utility function to generate example outline
            from utils.prompt_utils import generate_example_outline
            
            # Generate the example outline based on configuration
            example_outline = generate_example_outline(
                self.config.sizesections,
                self.config.sizeheadings
            )

            # Prepare prompt without explicit RAG context
            prompt = self.prompts.format_prompt(
                'outline',
                context_summary=self.context.get_context_summary(),
                keyword=keyword,
                articleaudience=self.config.articleaudience,
                sizeheadings=self.config.sizeheadings,
                articletype=self.config.articletype,
                articlelanguage=self.config.articlelanguage,
                sizesections=self.config.sizesections,
                example_outline=example_outline,
            )
            if web_context != "":
                prompt += f"\n\nFollow this Web context and use the data in it: {web_context}"

            # Use outline seed if seed control is enabled
            seed = self.config.outline_seed if self.config.enable_seed_control else None

            self.context.add_message("user", prompt)
            outline = self._generate_content(
                prompt,
                max_tokens=self.config.token_limits['outline'],
                generation_type="content_generation",
                seed=seed
            )

            # Display generated outline
            provider.success("Outline generated successfully")
            provider.debug(f"Outline content:\n{outline}")

            # Display token usage if enabled
            if self.config.enable_token_tracking:
                provider.token_usage(f"Outline generation - Tokens used: {self.context.count_message_tokens({'role': 'assistant', 'content': outline})}")

            # Don't add to context again - generate_completion already did that
            # Just update the article parts
            self.context.article_parts["outline"] = outline
            return outline
        except Exception as e:
            provider.error(f"Error generating outline: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return ""

    def generate_introduction(self, keyword: str, title: str, web_context:str = "") -> str:
        """Generates the article introduction."""
        try:
            provider.info(f"Generating introduction for: {title}")

            # Prepare prompt without explicit RAG context
            prompt = self.prompts.format_prompt(
                'introduction',
                context_summary=self.context.get_context_summary(),
                keyword=keyword,
                title=title,
                articlelanguage=self.config.articlelanguage,
                articleaudience=self.config.articleaudience,
                voicetone=self.config.voicetone,
                pointofview=self.config.pointofview,
                articletype=self.config.articletype,
                sizesections=self.config.sizesections
            )

            if web_context != "":
                prompt += f"\n\nFollow this Web context and use the data in it: {web_context}"

            # Use introduction seed if seed control is enabled
            seed = self.config.introduction_seed if self.config.enable_seed_control else None

            self.context.add_message("user", prompt)
            raw_text = self._generate_content(
                prompt,
                max_tokens=self.config.token_limits['introduction'],
                generation_type="content_generation",
                seed=seed
            )
            formatted_text = self._format_content(raw_text)

            # Display generated introduction
            provider.success("Introduction generated successfully")
            provider.debug(f"Introduction length: {len(raw_text)} characters")

            # Display token usage if enabled
            if self.config.enable_token_tracking:
                provider.token_usage(f"Introduction generation - Tokens used: {self.context.count_message_tokens({'role': 'assistant', 'content': formatted_text})}")

            # Don't add to context again - generate_completion already did that
            # Just update the article parts
            self.context.article_parts["introduction"] = formatted_text
            return formatted_text
        except Exception as e:
            provider.error(f"Error generating introduction: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return ""

    # def generate_paragraph(self, keyword: str, subtitle: str, current_paragraph: int = 1, paragraphs_per_section: int = None, section_number: int = 1, total_sections: int = 1, section_points: List[str] = None, web_context:str = "") -> str:
    #     """Generates a paragraph for a specific subtitle with explicit positioning."""
    #     try:
    #         provider.debug(f"Generating paragraph {current_paragraph}/{paragraphs_per_section} for: {subtitle}")

    #         # Set defaults if not provided
    #         if paragraphs_per_section is None:
    #             paragraphs_per_section = self.config.paragraphs_per_section
                
    #         # Default empty list for section points if none provided
    #         if section_points is None:
    #             section_points = []
                
    #         # Distribute points across paragraphs if there are enough points
    #         if section_points and len(section_points) > 1 and paragraphs_per_section > 1:
    #             points_per_paragraph = max(1, len(section_points) // paragraphs_per_section)
    #             start_idx = (current_paragraph - 1) * points_per_paragraph
    #             end_idx = min(start_idx + points_per_paragraph, len(section_points))
                
    #             # Points for this specific paragraph
    #             current_points = section_points[start_idx:end_idx]
                
    #             # Format current points as a string
    #             current_points_str = "\n".join([f"- {point}" for point in current_points]) if current_points else "- General information about this topic"
    #         else:
    #             current_points_str = "- Cover relevant information for this paragraph"
            
    #         # Format all points as a string for overall context
    #         all_points_str = "\n".join([f"- {point}" for point in section_points]) if section_points else "- General information about this topic"

    #         # Calculate available tokens for this paragraph
    #         available_tokens = min(
    #             self.config.token_limits['section'] // paragraphs_per_section,
    #             self.context.get_available_tokens() - 100  # Leave some padding
    #         )

    #         # Adjust flow instruction based on position in section
    #         # Flow instruction logic for paragraph positioning
    #         flow_instruction = self.get_flow_instruction(current_paragraph,paragraphs_per_section)

    #         # Prepare prompt with explicit paragraph positioning
    #         prompt = self.prompts.format_prompt(
    #             'paragraph',
    #             context_summary=self.context.get_context_summary(),
    #             keyword=keyword,
    #             subtitle=subtitle,
    #             articlelanguage=self.config.articlelanguage,
    #             articleaudience=self.config.articleaudience,
    #             voicetone=self.config.voicetone,
    #             pointofview=self.config.pointofview,
    #             current_paragraph=current_paragraph,
    #             paragraphs_per_section=paragraphs_per_section,
    #             section_number=section_number,
    #             total_sections=total_sections,
    #             all_points=all_points_str,
    #             current_points=current_points_str,
    #             flow_instruction=flow_instruction,
    #             articletype=self.config.articletype,
    #         )

    #         if web_context != "":
    #             prompt += f"\n\nFollow this Web context and use the data in it: {web_context}"

    #         # Use paragraph seed if seed control is enabled
    #         seed = self.config.paragraph_seed if self.config.enable_seed_control else None

    #         self.context.add_message("user", prompt)
    #         raw_text = self._generate_content(
    #             prompt,
    #             max_tokens=available_tokens,
    #             generation_type="content_generation",
    #             seed=seed,
    #             allowed= "all"
    #         )

    #         # Log the content length for debugging
    #         provider.debug(f"Generated paragraph for '{subtitle}': {len(raw_text)} characters")

    #         # Display token usage if enabled
    #         if self.config.enable_token_tracking:
    #             provider.token_usage(f"Paragraph generation for '{subtitle}' - Tokens used: {self.context.count_message_tokens({'role': 'assistant', 'content': raw_text})}")

    #         formatted_text = self._format_content(raw_text)
    #         # Don't add to context again - generate_completion already did that
    #         return formatted_text
    #     except Exception as e:
    #         provider.error(f"Error generating paragraph: {str(e)}")
    #         provider.error(f"Stack trace:\n{traceback.format_exc()}")
    #         return ""

    def generate_conclusion(self, keyword: str, title: str, web_context:str = "") -> str:
        """Generates the article conclusion."""
        try:
            provider.info(f"Generating conclusion for: {title}")

            # Prepare prompt without explicit RAG context
            prompt = self.prompts.format_prompt(
                'conclusion',
                context_summary=self.context.get_context_summary(),
                keyword=keyword,
                title=title,
                articleaudience=self.config.articleaudience,
                articlelanguage=self.config.articlelanguage,
                voicetone=self.config.voicetone,
                pointofview=self.config.pointofview,
                articletype=self.config.articletype,
            )

            if web_context != "":
                prompt += f"\n\nFollow this Web context and use the data in it: {web_context}"

            # Use conclusion seed if seed control is enabled
            seed = self.config.conclusion_seed if self.config.enable_seed_control else None

            self.context.add_message("user", prompt)
            raw_text = self._generate_content(
                prompt,
                max_tokens=self.config.token_limits['conclusion'],
                generation_type="content_generation",
                seed=seed
            )

            # Log the content length for debugging
            provider.debug(f"Generated conclusion length: {len(raw_text)} characters")

            # Display token usage if enabled
            if self.config.enable_token_tracking:
                provider.token_usage(f"Conclusion generation - Tokens used: {self.context.count_message_tokens({'role': 'assistant', 'content': raw_text})}")

            formatted_text = self._format_content(raw_text)
            provider.success("Conclusion generated successfully")

            # Don't add to context again - generate_completion already did that
            # Just update the article parts
            self.context.article_parts["conclusion"] = formatted_text
            return formatted_text
        except Exception as e:
            provider.error(f"Error generating conclusion: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return ""

    def generate_faq(self, keyword: str, web_context:str = "") -> Optional[str]:
        """Generates FAQ section for the article."""
        if not self.config.add_faq_into_article:
            provider.info("FAQ generation is disabled in configuration")
            return None

        try:
            provider.info(f"Generating FAQ section for: {keyword}")

            tone = getattr(self.config, 'voicetone', 'neutral')

            prompt = self.prompts.format_prompt(
                'faq',
                context_summary=self.context.get_context_summary(),
                keyword=keyword,
                num_questions=2,
                articlelanguage=self.config.articlelanguage,
                articleaudience=self.config.articleaudience,
                voicetone=tone,
                pointofview=self.config.pointofview
            )

            if web_context != "":
                prompt += f"\n\nFollow this Web context and use the data in it: {web_context}"

            # Use FAQ seed if seed control is enabled
            seed = self.config.faq_seed if self.config.enable_seed_control else None

            self.context.add_message("user", prompt)
            raw_text = self._generate_content(
                prompt,
                max_tokens=self.config.token_limits['faq'],
                generation_type="faq_generation",
                seed=seed
            )

            provider.debug(f"Raw FAQ response:\n{raw_text}")

            # Display token usage if enabled
            if self.config.enable_token_tracking:
                provider.token_usage(f"FAQ generation - Tokens used: {self.context.count_message_tokens({'role': 'assistant', 'content': raw_text})}")

            # Clean up the response to extract valid JSON
            # Find the first '[' and last ']' to extract the JSON array
            start_idx = raw_text.find('[')
            end_idx = raw_text.rfind(']')

            if start_idx == -1 or end_idx == -1:
                provider.error(f"Could not find JSON array markers in response. Response was:\n{raw_text}")
                return None

            json_str = raw_text[start_idx:end_idx + 1]
            # Clean up common formatting issues
            json_str = json_str.replace('\n', '')
            json_str = json_str.replace('  ', ' ')

            provider.debug(f"Cleaned JSON string:\n{json_str}")

            try:
                faq_data = json.loads(json_str)

                # Validate the structure
                if not isinstance(faq_data, list):
                    provider.error("FAQ data is not a list")
                    return None

                # Format FAQ content
                faq_content = []
                for qa in faq_data:
                    if isinstance(qa, dict) and "question" in qa and "answer" in qa:
                        faq_content.extend([
                            f'<!-- wp:heading {{"level":3}} -->',
                            f'<h3>{qa["question"]}</h3>',
                            '<!-- /wp:heading -->',
                            '<!-- wp:paragraph -->',
                            f'<p>{qa["answer"]}</p>',
                            '<!-- /wp:paragraph -->'
                        ])

                if faq_content:
                    formatted_faq = "\n\n".join(faq_content)
                    provider.success(f"Generated FAQ section with {len(faq_data)} Q&A pairs")
                    return formatted_faq
                else:
                    provider.error("No valid Q&A pairs found in FAQ data")
                    return None

            except json.JSONDecodeError as e:
                provider.error(f"Error parsing FAQ JSON: {str(e)}")
                return None

        except Exception as e:
            provider.error(f"Error generating FAQ section: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return None

    def generate_article_summary(
        self,
        keyword: str,
        article_dict: Dict[str, str],
    ) -> str:
        """
        Generate a comprehensive summary of the article.

        Uses a large context window model if configured, with chunking for very large articles.
        """
        logger.info("Generating article summary...")

        try:
            # Determine if we should use a separate model for summary generation
            use_separate_model = (
                hasattr(self.config, 'enable_separate_summary_model') and
                self.config.enable_separate_summary_model and
                hasattr(self.config, 'summary_keynotes_model') and
                self.config.summary_keynotes_model
            )

            # Get the chunk size from config or use default
            chunk_size = getattr(self.config, 'summary_chunk_size', 8000)

            # Chunk the article if needed
            article_chunks = chunk_article_for_processing(article_dict, chunk_size=chunk_size)
            logger.info(f"Article split into {len(article_chunks)} chunks for summary generation")

            chunk_results = []

            for i, chunk in enumerate(article_chunks):
                logger.info(f"Processing chunk {i+1}/{len(article_chunks)} for summary")

                # Compile the article content for summarization
                full_content = (
                    f"Title: {chunk.get('title', '')}\n\n"
                    f"Introduction: {chunk.get('introduction', '')}\n\n"
                    f"Main Content:\n{chr(10).join(chunk.get('sections', []))}\n\n"
                    f"Conclusion: {chunk.get('conclusion', '')}"
                )

                # Format the summary prompt
                prompt = self.prompts.summarize.format(
                    keyword=keyword,
                    articleaudience=getattr(self.context, 'articleaudience', 'General'),
                    article_content=full_content,
                )

                # Use summary seed if seed control is enabled
                seed = (
                    self.config.summary_seed
                    if hasattr(self.config, 'enable_seed_control') and self.config.enable_seed_control and hasattr(self.config, 'summary_seed')
                    else None
                )

                # Generate summary with the appropriate model
                if use_separate_model and self.config.use_openrouter:
                    logger.info(f"Using separate model for summary generation: {self.config.summary_keynotes_model}")

                    # Create messages for the API call
                    messages = [
                        {"role": "system", "content": "You are an expert content writer specializing in creating comprehensive article summaries."},
                        {"role": "user", "content": prompt}
                    ]

                    # Make the API call
                    # Get temperature and other parameters from config or use defaults
                    temperature = getattr(self.config, "summary_temperature", 0.7)
                    summary_top_p = getattr(self.config, "summary_top_p", 1.0)
                    summary_frequency_penalty = getattr(self.config, "summary_frequency_penalty", 0.0)
                    summary_presence_penalty = getattr(self.config, "summary_presence_penalty", 0.0)

                    response = make_openrouter_api_call(
                        messages=messages,
                        model=self.config.summary_keynotes_model,
                        api_key=self.config.openrouter_api_key,
                        site_url=self.config.openrouter_site_url,
                        site_name=self.config.openrouter_site_name,
                        temperature=temperature,
                        max_tokens=getattr(self.config, 'summary_max_tokens', 800),
                        seed=seed,
                        top_p=summary_top_p,
                        frequency_penalty=summary_frequency_penalty,
                        presence_penalty=summary_presence_penalty
                    )

                    chunk_summary = response.choices[0].message.content.strip()
                else:
                    # Use the standard generate_completion function
                    # Determine which model to use based on whether OpenRouter is enabled
                    model_to_use = self.config.openrouter_model if (hasattr(self.config, 'use_openrouter') and self.config.use_openrouter and self.config.openrouter_api_key) else self.config.openai_model
                    
                    chunk_summary = generate_completion(
                        prompt=prompt,
                        model=model_to_use,
                        temperature=getattr(self.config, "summary_temperature", 0.7),
                        max_tokens=getattr(self.config, 'summary_max_tokens', 800),
                        article_context=self.context,
                        seed=seed,
                    )

                if chunk_summary:
                    chunk_results.append(chunk_summary)

            # Combine results from all chunks
            if not chunk_results:
                logger.warning("No summary was generated from any chunk")
                return ""

            # Use the LLM to combine chunks if there are multiple chunks
            if len(chunk_results) > 1:
                logger.info("Using LLM to combine summary chunks")
                summary = combine_chunk_results_with_llm(chunk_results, self.context, self.prompts.summary_combine, is_summary=True)
            else:
                summary = chunk_results[0]

            logger.success(f"Generated article summary ({len(summary.split())} words)")
            return summary.strip()

        except Exception as e:
            logger.error(f"Error generating article summary: {str(e)}")
            # Return empty string on error rather than raising
            return ""

    def _generate_content(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        generation_type: str = "content_generation",
        seed: Optional[int] = None,
        allowed:str = "controlled"
    ) -> str:
        """Send prompt to OpenAI API or OpenRouter and get a response."""
        try:
            # Log the current state of the context before generating content
            provider.info(f"CONTENT_GEN: Generating content for {generation_type}")
            provider.info(f"CONTENT_GEN: Context before generation: {len(self.context.messages)} messages")

            # Log message roles for debugging
            message_roles = [msg["role"] for msg in self.context.messages]
            provider.debug(f"CONTENT_GEN: Current message roles: {message_roles}")

            # Apply rate limiting if enabled - use the object properly
            if hasattr(self.config, 'enable_rate_limiting') and self.config.enable_rate_limiting and openai_rate_limiter:
                # Use wait_until_ready instead of direct function call
                openai_rate_limiter.wait_until_ready()

            # Get generation parameters based on the type
            temperature = temperature or getattr(self.config, f"{generation_type}_temperature", 1.0)
            top_p = top_p or getattr(self.config, f"{generation_type}_top_p", 1.0)
            freq_penalty = frequency_penalty or getattr(self.config, f"{generation_type}_frequency_penalty", 0.0)
            presence_penalty = presence_penalty or getattr(self.config, f"{generation_type}_presence_penalty", 0.0)

            # RAG context is now handled at the article context level,
            # not for individual prompts

            # Use the unified generate_completion function that supports both OpenAI and OpenRouter
            provider.info(f"CONTENT_GEN: Calling generate_completion for {generation_type}")

            # Determine which model to use based on whether OpenRouter is enabled
            model_to_use = self.config.openrouter_model if (hasattr(self.config, 'use_openrouter') and self.config.use_openrouter and self.config.openrouter_api_key) else self.config.openai_model
            
            content = generate_completion(
                prompt=prompt,
                model=model_to_use,
                temperature=temperature,
                max_tokens=max_tokens or 2048,
                article_context=self.context,
                top_p=top_p,
                frequency_penalty=freq_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                allowed= allowed
            )

            # Log the current state of the context after generating content
            provider.info(f"CONTENT_GEN: Context after generation: {len(self.context.messages)} messages")

            # Log message roles for debugging
            message_roles = [msg["role"] for msg in self.context.messages]
            provider.debug(f"CONTENT_GEN: Updated message roles: {message_roles}")

            return content.replace('```html', ' ').replace('```', ' ').strip()
        except Exception as e:
            provider.error(f"Error generating content: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return ""

    def _extract_query_from_prompt(self, prompt):
        """Extract a search query from the prompt."""
        # Simple extraction - get the first sentence or keyword mention
        lines = prompt.split('\n')
        for line in lines:
            if "keyword:" in line.lower():
                return line.split("keyword:")[1].strip()

        # If no keyword found, use the first sentence
        first_sentence = prompt.split('.')[0]
        if len(first_sentence) > 10:  # Ensure it's not too short
            return first_sentence

        # Default to the first 100 characters
        return prompt[:100].strip()

    def _format_content(self, raw_text: str) -> str:
        """Format raw text content for WordPress."""
        try:
            # Convert markdown bold (double asterisks) to WordPress bold tags
            text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', raw_text)

            # Convert markdown italic (single asterisk) to WordPress italic tags
            text = re.sub(r'\*([^\*]+)\*', r'<em>\1</em>', text)

            # Remove any remaining markdown formatting
            text = re.sub(r'#+\s+', '', text, flags=re.MULTILINE)  # Remove markdown headings

            # Clean up whitespace
            text = text.strip()
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)

            # Format paragraphs for WordPress
            paragraphs = text.split('\n\n')
            formatted_paragraphs = []
            for p in paragraphs:
                if p.strip():
                    formatted_paragraphs.append(f'<!-- wp:paragraph -->\n<p>{p.strip()}</p>\n<!-- /wp:paragraph -->')

            return '\n\n'.join(formatted_paragraphs)
        except Exception as e:
            provider.error(f"Error formatting content: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return raw_text

    def clear_context(self) -> None:
        """Clear the conversation context."""
        try:
            # Reset the context
            self.context.clear_messages()

            # Log the API provider being used again for clarity
            if self.config.use_openrouter and self.config.openrouter_api_key:
                logger.info(f"Using OpenRouter for API calls with site: {self.config.openrouter_site_name or 'None'}")
                provider.debug(f"OpenRouter model mapping active for '{self.config.openai_model}'")
            else:
                logger.info(f"Using OpenAI direct API with model: {self.config.openai_model}")

            provider.debug("Cleared conversation context")
        except Exception as e:
            provider.error(f"Error clearing context: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            
    

    # Flow instruction logic for paragraph positioning, supporting all possible HTML output formats with optional structure
    def get_flow_instruction(self,current_paragraph, paragraphs_per_section):
        if current_paragraph == 1:
            return (
                "This is the first paragraph for this section. Introduce the topic clearly using a <strong> tag for the "
                "primary keyword, setting the stage for subsequent paragraphs with an engaging example. Select the most "
                "effective HTML structure (<ul><li> for tips, <ol><li> for steps, <table> for comparisons, or simple text "
                "with <strong> and <em>) based on the best way to represent the information, ensuring a smooth entry into the "
                "section with high scannability."
            )
        elif current_paragraph == paragraphs_per_section:
            return (
                "This is the last paragraph for this section. Provide a concise summary or a smooth transition to the next "
                "section, reinforcing the section’s key points with a <strong> tag for emphasis. Choose the most effective "
                "HTML structure (<table> for summarizing key takeaways, <ul><li> for key points, <ol><li> for final steps, "
                "or text with <em> for subtle emphasis) to clearly convey the message and enhance SEO performance."
            )
        else:
            return (
                f"This is paragraph {current_paragraph} of {paragraphs_per_section}. Ensure a smooth transition from previous "
                "content, develop the topic further with specific details or examples, and maintain coherence. Select the most "
                "effective HTML structure (<ul><li> for actionable tips, <ol><li> for sequential steps, <table> for "
                "comparisons, or text with <strong> and <em>) based on the best way to represent the information, enhancing "
                "scannability and engagement."
            )

    def generate_paragraph(self, keyword: str, subtitle: str, current_paragraph: int = 1, paragraphs_per_section: int = None, section_number: int = 1, total_sections: int = 1, section_points: List[str] = None, web_context: str = "") -> str:
        """Generates a paragraph with heading for a specific subtitle in a single API call."""
        try:
            provider.debug(f"Generating paragraph {current_paragraph}/{paragraphs_per_section} with heading for: {subtitle}")

            # Set defaults if not provided
            if paragraphs_per_section is None:
                paragraphs_per_section = self.config.paragraphs_per_section
                
            # Default empty list for section points if none provided
            if section_points is None:
                section_points = []
                
            # Distribute points across paragraphs if there are enough points
            if section_points and len(section_points) > 1 and paragraphs_per_section > 1:
                points_per_paragraph = max(1, len(section_points) // paragraphs_per_section)
                start_idx = (current_paragraph - 1) * points_per_paragraph
                end_idx = min(start_idx + points_per_paragraph, len(section_points))
                
                # Points for this specific paragraph
                current_points = section_points[start_idx:end_idx]
                
                # Format current points as a string
                current_points_str = "\n".join([f"- {point}" for point in current_points]) if current_points else "- General information about this topic"
            else:
                current_points_str = "- Cover relevant information for this paragraph"
            
            # Format all points as a string for overall context
            all_points_str = "\n".join([f"- {point}" for point in section_points]) if section_points else "- General information about this topic"

            # Adjust flow instruction based on position in section
            # Flow instruction logic for paragraph positioning
            flow_instruction = self.get_flow_instruction(current_paragraph,paragraphs_per_section)
                    
          # Prepare prompt for paragraph with heading
            prompt = self.prompts.format_prompt(
                'paragraph',
                context_summary=self.context.get_context_summary(),
                keyword=keyword,
                subtitle=subtitle,
                articlelanguage=self.config.articlelanguage,
                articleaudience=self.config.articleaudience,
                voicetone=self.config.voicetone,
                pointofview=self.config.pointofview,
                current_paragraph=current_paragraph,
                paragraphs_per_section=paragraphs_per_section,
                section_number=section_number,
                total_sections=total_sections,
                all_points=all_points_str,
                current_points=current_points_str,
                flow_instruction=flow_instruction,
                articletype=self.config.articletype,
            )

            if web_context != "":
                prompt += f"\n\nFollow this Web context and use the data in it: {web_context}"

            # Use paragraph seed if seed control is enabled
            seed = self.config.paragraph_seed if self.config.enable_seed_control else None

            # Calculate available tokens for this paragraph
            available_tokens = min(
                self.config.token_limits['section'] // paragraphs_per_section,
                self.context.get_available_tokens() - 100  # Leave some padding
            )

            self.context.add_message("user", prompt)
            response = self._generate_content(
                prompt,
                max_tokens=available_tokens,
                generation_type="content_generation",
                seed=seed,
                allowed="all"
            )
            
       
            # Format with HTML tags
            formatted_paragraph = f'<p>{response}</p>'
            return formatted_paragraph
                
        except Exception as e:
            provider.error(f"Error generating paragraph with heading: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return f'<h4>About {keyword}</h4>\n\n<p>Information about {subtitle} related to {keyword}.</p>'