"""
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

Meta description and excerpt generation handler for article generation.
Handles both SEO meta descriptions and WordPress excerpts.
"""

from typing import Optional
import traceback
import time
from utils.rich_provider import provider
from openai import APIError, RateLimitError
import openai
from utils.ai_utils import generate_completion, make_openrouter_api_call

# Rate limiting configuration
MAX_RETRIES = 5
INITIAL_DELAY = 1  # Initial delay in seconds
MAX_DELAY = 60     # Maximum delay in seconds

class MetaHandler:
    """Handles generation of meta descriptions and excerpts for articles."""

    def __init__(self, config, openai_client, prompts):
        """Initialize MetaHandler with configuration and OpenAI client.

        Args:
            config: Configuration object containing meta settings
            openai_client: OpenAI client for text generation
            prompts: Prompts object containing meta description templates
        """
        try:
            self.config = config
            self.openai_client = openai_client
            self.prompts = prompts

            # Log API provider being used
            if hasattr(config, 'use_openrouter') and config.use_openrouter and config.openrouter_api_key:
                provider.debug(f"MetaHandler using OpenRouter with site: {config.openrouter_site_name}")
            else:
                provider.debug(f"MetaHandler using OpenAI direct API with model: {config.openai_model}")

            provider.debug("MetaHandler initialized")
        except Exception as e:
            provider.error(f"Error initializing MetaHandler: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            raise

    def _generate_with_constraints(self, prompt: str, context=None, seed: Optional[int] = None) -> str:
        """Generate text with length constraints for meta descriptions and excerpts.

        Args:
            prompt: Prompt for text generation
            context: Optional ArticleContext for token tracking
            seed: Optional seed for deterministic generation

        Returns:
            Generated text within constraints
        """
        try:
            MAX_RETRIES = 3
            MAX_DELAY = 20

            # Use the unified generate_completion function for OpenRouter compatibility
            messages = [
                {"role": "system", "content": self.prompts.system_message},
                {"role": "user", "content": prompt}
            ]

            # If we have a context, add the user message to the existing context
            if context:
                # Add the user message to the context instead of replacing all messages
                context.add_message("user", prompt)

                # Log what we're doing
                provider.debug(f"META: Using existing context with {len(context.messages)} messages")

                # Determine which model to use based on whether OpenRouter is enabled
                model_to_use = self.config.openrouter_model if (hasattr(self.config, 'use_openrouter') and self.config.use_openrouter and self.config.openrouter_api_key) else self.config.openai_model
                
                generated_text = generate_completion(
                    prompt=prompt,
                    model=model_to_use,
                    temperature=self.config.meta_description_temperature,
                    max_tokens=200,  # Allow enough tokens for meta description
                    article_context=context,
                    top_p=self.config.meta_description_top_p,
                    frequency_penalty=self.config.meta_description_frequency_penalty,
                    presence_penalty=self.config.meta_description_presence_penalty,
                    seed=seed if self.config.enable_seed_control else None
                )
            else:
                # If no context is provided, use OpenAI client directly but check for OpenRouter
                if hasattr(self.config, 'use_openrouter') and self.config.use_openrouter and self.config.openrouter_api_key:
                    # Get model name based on mapping if needed
                    model_to_use = self.config.openai_model
                    if self.config.openrouter_models:
                        for key, full_model_id in self.config.openrouter_models.items():
                            if key.lower() in model_to_use.lower():
                                model_to_use = full_model_id
                                break

                    # Use OpenRouter API
                    response = make_openrouter_api_call(
                        messages=messages,
                        model=model_to_use,
                        api_key=self.config.openrouter_api_key,
                        site_url=self.config.openrouter_site_url or "https://example.com",
                        site_name=self.config.openrouter_site_name or "AI Article Generator",
                        temperature=self.config.meta_description_temperature,
                        max_tokens=200,
                        seed=seed if self.config.enable_seed_control else None,
                        top_p=self.config.meta_description_top_p,
                        frequency_penalty=self.config.meta_description_frequency_penalty,
                        presence_penalty=self.config.meta_description_presence_penalty
                    )
                    generated_text = response.choices[0].message.content
                else:
                    # Use OpenAI API directly
                    api_params = {
                        "model": self.config.openai_model,
                        "messages": messages,
                        "temperature": self.config.meta_description_temperature,
                        "max_tokens": 200,
                        "top_p": self.config.meta_description_top_p,
                        "frequency_penalty": self.config.meta_description_frequency_penalty,
                        "presence_penalty": self.config.meta_description_presence_penalty
                    }

                    # Add seed if seed control is enabled and a seed is provided
                    if self.config.enable_seed_control and seed is not None:
                        api_params["seed"] = seed

                    # Make API call
                    response = self.openai_client.chat.completions.create(**api_params)
                    generated_text = response.choices[0].message.content.strip()

                    # Track token usage if enabled
                    if self.config.enable_token_tracking:
                        provider.token_usage(f"Meta generation - Request tokens: {response.usage.prompt_tokens}")
                        provider.token_usage(f"Meta generation - Response tokens: {response.usage.completion_tokens}")
                        provider.token_usage(f"Meta generation - Total tokens: {response.usage.total_tokens}")

            # Ensure the text is within the constraints
            if len(generated_text) < self.config.meta_description_min_length:
                provider.warning(f"Generated text too short ({len(generated_text)} chars), minimum is {self.config.meta_description_min_length}")
                # Return it anyway, better than nothing

            if len(generated_text) > self.config.meta_description_max_length:
                provider.warning(f"Generated text too long ({len(generated_text)} chars), truncating to {self.config.meta_description_max_length}")
                generated_text = generated_text[:self.config.meta_description_max_length]

                # Try to find a sentence boundary for cleaner truncation
                last_period = generated_text.rfind('.')
                if last_period > self.config.meta_description_min_length:
                    generated_text = generated_text[:last_period + 1]

            return generated_text

        except Exception as e:
            provider.error(f"Error generating meta text: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return ""

    def generate_meta_description(self, keyword: str, context=None, article_type: Optional[str] = None, article_audience: Optional[str] = None) -> str:
        """Generate SEO meta description for the article.

        Args:
            keyword: Main keyword for the article
            context: Optional ArticleContext for token tracking
            article_type: Type of article being generated
            article_audience: Target audience for the article

        Returns:
            Generated meta description
        """
        try:
            provider.info(f"Generating meta description for keyword: {keyword}")
            prompt = self.prompts.meta_description.format(
                keyword=keyword,
                articletype=article_type or self.config.articletype,
                articlelanguage=self.config.articlelanguage,
                voicetone=self.config.voicetone,
                articleaudience=article_audience or self.config.articleaudience
            )

            # Use meta_description seed if seed control is enabled
            seed = self.config.meta_description_seed if self.config.enable_seed_control else None

            result = self._generate_with_constraints(prompt, context, seed)
            if result:
                provider.success(f"Generated meta description ({len(result)} chars)")
            return result
        except Exception as e:
            provider.error(f"Error in generate_meta_description: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return ""

    def generate_wordpress_excerpt(self, keyword: str, context=None, article_type: Optional[str] = None, article_audience: Optional[str] = None) -> str:
        """Generate WordPress excerpt for the article.

        Args:
            keyword: Main keyword for the article
            context: Optional ArticleContext for token tracking
            article_type: Type of article being generated
            article_audience: Target audience for the article

        Returns:
            Generated WordPress excerpt
        """
        try:
            provider.info(f"Generating WordPress excerpt for keyword: {keyword}")
            prompt = self.prompts.wordpress_excerpt.format(
                keyword=keyword,
                articletype=article_type or self.config.articletype,
                articlelanguage=self.config.articlelanguage,
                voicetone=self.config.voicetone,
                articleaudience=article_audience or self.config.articleaudience
            )

            # Use meta_description seed if seed control is enabled (same seed for consistency)
            seed = self.config.meta_description_seed if self.config.enable_seed_control else None

            result = self._generate_with_constraints(prompt, context, seed)
            if result:
                provider.success(f"Generated WordPress excerpt ({len(result)} chars)")
            return result
        except Exception as e:
            provider.error(f"Error in generate_wordpress_excerpt: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return ""