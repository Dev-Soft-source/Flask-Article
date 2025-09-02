"""
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

Meta description and excerpt generation handler for article generation.
Handles both SEO meta descriptions and WordPress excerpts.
"""

from typing import Optional
import time
from .logger import logger
from utils.rate_limiter import openai_rate_limiter

class MetaHandler:
    """Handles generation of meta descriptions and excerpts for articles."""

    def __init__(self, config, openai_client, prompts):
        """Initialize MetaHandler with configuration and OpenAI client.

        Args:
            config: Configuration object containing meta settings
            openai_client: OpenAI client for text generation
            prompts: Prompts object containing meta description templates
        """
        self.config = config
        self.openai_client = openai_client
        self.prompts = prompts
        logger.debug("MetaHandler initialized")

    def _generate_with_constraints(self, prompt: str, seed: Optional[int] = None) -> str:
        """Generate text with OpenAI while respecting length constraints.

        Args:
            prompt: The prompt to send to OpenAI
            seed: Optional seed for deterministic generation

        Returns:
            Generated text within length constraints
        """
        try:
            logger.debug("Generating meta text with OpenAI...")

            # Prepare API call parameters
            api_params = {
                "model": getattr(self.config, "openai_model", "gpt-4o-mini-2024-07-18"),  # Use openai_model instead of openai_engine
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.config.meta_description_temperature,
                "max_tokens": 200,  # Enough for meta description
                "top_p": self.config.meta_description_top_p,
                "frequency_penalty": self.config.meta_description_frequency_penalty,
                "presence_penalty": self.config.meta_description_presence_penalty
            }

            # Add seed if seed control is enabled and a seed is provided
            if self.config.enable_seed_control and seed is not None:
                api_params["seed"] = seed
                logger.debug(f"Using seed {seed} for meta text generation")

            # Use rate limiter if available
            if self.config.enable_rate_limiting and openai_rate_limiter:
                logger.debug("Using rate limiter for OpenAI API call")

                def make_api_call():
                    return self.openai_client.chat.completions.create(**api_params)

                response = openai_rate_limiter.execute_with_rate_limit(make_api_call)
            else:
                response = self.openai_client.chat.completions.create(**api_params)

            if response.choices:
                text = response.choices[0].message.content.strip()

                # Ensure text meets length constraints
                if len(text) > self.config.meta_description_max_length:
                    logger.debug(f"Trimming text from {len(text)} to {self.config.meta_description_max_length} chars")
                    text = text[:self.config.meta_description_max_length].rsplit(' ', 1)[0]
                elif len(text) < self.config.meta_description_min_length:
                    logger.warning(f"Generated meta text below minimum length: {len(text)} chars")

                logger.debug(f"Generated meta text ({len(text)} chars)")
                return text

        except Exception as e:
            logger.error(f"Error generating meta text: {str(e)}")
            return ""

    def generate_meta_description(self, keyword: str, article_type: Optional[str] = None, article_audience: Optional[str] = None) -> str:
        """Generate SEO meta description for the article.

        Args:
            keyword: Main keyword for the article
            article_type: Type of article being generated
            article_audience: Target audience for the article

        Returns:
            Generated meta description
        """
        logger.info(f"Generating meta description for keyword: {keyword}")
        prompt = self.prompts.meta_description.format(
            keyword=keyword,
            articletype=article_type or self.config.articletype,
            articlelanguage=self.config.articlelanguage,
            voicetone=self.config.voicetone,
            articleaudience=article_audience or self.config.articleaudience
        )

        # Use meta_description seed if seed control is enabled
        seed = self.config.meta_description_seed if self.config.enable_seed_control else None

        result = self._generate_with_constraints(prompt, seed)
        if result:
            logger.success(f"Generated meta description ({len(result)} chars)")
        return result

    def generate_wordpress_excerpt(self, keyword: str, article_type: Optional[str] = None, article_audience: Optional[str] = None) -> str:
        """Generate WordPress excerpt for the article.

        Args:
            keyword: Main keyword for the article
            article_type: Type of article being generated
            article_audience: Target audience for the article

        Returns:
            Generated WordPress excerpt
        """
        logger.info(f"Generating WordPress excerpt for keyword: {keyword}")
        prompt = self.prompts.wordpress_excerpt.format(
            keyword=keyword,
            articletype=article_type or self.config.articletype,
            articlelanguage=self.config.articlelanguage,
            voicetone=self.config.voicetone,
            articleaudience=article_audience or self.config.articleaudience
        )

        # Use meta_description seed for WordPress excerpt as well
        # We could add a separate seed for this if needed in the future
        seed = self.config.meta_description_seed if self.config.enable_seed_control else None

        result = self._generate_with_constraints(prompt, seed)
        if result:
            logger.success(f"Generated WordPress excerpt ({len(result)} chars)")
        return result