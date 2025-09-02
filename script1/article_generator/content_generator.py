# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import re
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Dict, List, Optional, Tuple
import time
from dataclasses import dataclass
import sys
import os
import traceback
from utils.config import Config
from utils.prompts_config import Prompts
from article_generator.logger import logger
from utils.rate_limiter import openai_rate_limiter, initialize_rate_limiters, RateLimitConfig
from utils.rate_limit_error import RateLimitError
import json
from datetime import datetime
import requests
import time
from utils.error_utils import ErrorHandler


# Initialize error handler
error_handler = ErrorHandler()

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize rate limiter if not already initialized
if openai_rate_limiter is None:
    logger.info("Initializing OpenAI rate limiter...")
    initialize_rate_limiters(
        openai_config=RateLimitConfig(rpm=60, rpd=10000, enabled=True)
    )

@dataclass
class ArticleContext:
    """Context for article generation, containing settings and state."""

    config: Config  # Config object
    openai_engine: str
    max_context_window_tokens: int
    token_padding: int
    track_token_usage: bool
    warn_token_threshold: float
    section_token_limit: int
    paragraphs_per_section: int
    min_paragraph_tokens: int
    max_paragraph_tokens: int
    size_headings: int
    size_sections: int
    articletype: str
    articlelanguage: str
    voicetone: str
    pointofview: str
    articleaudience: str = "General"  # Default to General audience
    openrouter_model: str = None  # OpenRouter model to use when OpenRouter is enabled

    # Token tracking state
    total_tokens_used: int = 0
    completion_tokens: int = 0
    prompt_tokens: int = 0

    # Keyword for the article
    keyword: str = ""

    def __init__(
        self,
        config: Config,
        openai_engine: str,
        max_context_window_tokens: int,
        token_padding: int,
        track_token_usage: bool,
        warn_token_threshold: float,
        section_token_limit: int,
        paragraphs_per_section: int,
        min_paragraph_tokens: int,
        max_paragraph_tokens: int,
        size_headings: int,
        size_sections: int,
        articletype: str,
        articlelanguage: str,
        voicetone: str,
        pointofview: str,
        articleaudience: str = "General",
    ):
        """Initialize the context with the given parameters."""
        self.config = config
        self.openai_engine = openai_engine
        self.max_context_window_tokens = max_context_window_tokens
        self.token_padding = token_padding
        self.track_token_usage = track_token_usage
        self.warn_token_threshold = warn_token_threshold
        self.section_token_limit = section_token_limit
        self.paragraphs_per_section = paragraphs_per_section
        self.min_paragraph_tokens = min_paragraph_tokens
        self.max_paragraph_tokens = max_paragraph_tokens
        self.size_headings = size_headings
        self.size_sections = size_sections
        self.articletype = articletype
        self.articlelanguage = articlelanguage
        self.voicetone = voicetone
        self.pointofview = pointofview
        self.articleaudience = articleaudience
        self.openrouter_model = config.openrouter_model if hasattr(config, 'openrouter_model') else None

        # Initialize token tracking state
        self.total_tokens_used = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0

        # Initialize tokenizer
        try:
            import tiktoken
            self.encoding = tiktoken.encoding_for_model(self.openai_engine)
        except Exception as e:
            error_handler.handle_error(e, severity="error")
            # Fallback to a common tokenizer if specific model encoding fails
            try:
                import tiktoken
                self.encoding = tiktoken.get_encoding("cl100k_base")
                logger.warning(f"Using fallback tokenizer cl100k_base")
            except Exception as e2:
                logger.error(f"Failed to initialize fallback tokenizer: {str(e2)}")
                # Create a dummy encoding method as last resort
                class DummyEncoding:
                    def encode(self, text):
                        return [0] * (len(text) // 4)  # Rough approximation
                self.encoding = DummyEncoding()
                logger.error("Using emergency dummy tokenizer")

        # Initialize with system message
        system_msg = {
            "role": "system",
            "content": """You are an expert content writer creating cohesive, engaging articles.
            Maintain consistent tone, style, and narrative flow throughout the piece.
            Each response should build upon previous content while adding new value.
            Focus on clarity, accuracy, and engaging storytelling.""",
        }

        # Add system message and count its tokens
        self.messages = [system_msg]
        self.total_tokens = self.count_message_tokens(system_msg)

        # Initialize article parts
        self.article_parts = {
            "title": None,
            "outline": None,
            "introduction": None,
            "sections": [],
            "conclusion": None,
        }

        logger.success("Article context initialized")
        if self.track_token_usage:
            logger.debug(f"Token usage - Initial: {self.total_tokens}")

    def __post_init__(self):
        """Original post-init method, now just a pass-through as initialization is handled in __init__"""
        pass

    def update_token_usage(self, usage: Dict[str, int]) -> None:
        """Update token usage statistics."""
        if not self.track_token_usage:
            return

        # Handle both dictionary and CompletionUsage object
        if hasattr(usage, "completion_tokens"):
            # New OpenAI API returns object with attributes
            self.completion_tokens += usage.completion_tokens
            self.prompt_tokens += usage.prompt_tokens
        else:
            # Fallback for dictionary format
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.prompt_tokens += usage.get("prompt_tokens", 0)

        self.total_tokens_used = self.completion_tokens + self.prompt_tokens

        # Check if we're approaching the context window limit
        if self.total_tokens_used >= (
            self.max_context_window_tokens * self.warn_token_threshold
        ):
            logger.warning(
                f"Approaching token limit. Used {self.total_tokens_used} tokens "
                f"({(self.total_tokens_used/self.max_context_window_tokens)*100:.1f}% of max)"
            )

    def count_message_tokens(self, message: Dict[str, str]) -> int:
        """Count tokens in a message"""
        try:
            # Check if encoding exists, initialize it if not
            if not hasattr(self, 'encoding') or self.encoding is None:
                logger.warning("Tokenizer not initialized in count_message_tokens, trying to create it")
                try:
                    import tiktoken
                    self.encoding = tiktoken.encoding_for_model(self.openai_engine)
                except Exception as e:
                    logger.error(f"Failed to initialize tokenizer: {str(e)}")
                    try:
                        import tiktoken
                        self.encoding = tiktoken.get_encoding("cl100k_base")
                        logger.warning("Using fallback cl100k_base tokenizer")
                    except Exception as e2:
                        error_handler.handle_error(e2, severity="error")
                        # Create an emergency encoding method
                        logger.error("Using emergency token counting approximation")
                        return len(message["content"]) // 4  # Rough approximation (4 chars ≈ 1 token)

            return len(self.encoding.encode(message["content"])) + 4  # 4 tokens for message format
        except Exception as e:
            error_handler.handle_error(e, severity="error")
            # Emergency fallback
            return len(message["content"]) // 4  # Rough approximation (4 chars ≈ 1 token)

    def get_available_tokens(self) -> int:
        """Get number of tokens available in the context window"""
        return self.max_context_window_tokens - self.total_tokens - self.token_padding

    def would_exceed_limit(self, new_content: str) -> Tuple[bool, int]:
        """
        Check if adding new content would exceed token limits

        Args:
            new_content (str): Content to be added
        Returns:
            Tuple[bool, int]: (would exceed, tokens needed)
        """
        try:
            tokens_needed = len(self.encoding.encode(new_content))

            # Check if it would exceed the context window
            if (
                self.total_tokens + tokens_needed
                > self.max_context_window_tokens - self.token_padding
            ):
                return True, tokens_needed

            return False, tokens_needed
        except Exception as e:
            error_handler.handle_error(e, severity="error")
            return True, 0

    def prune_oldest_message(self) -> int:
        """
        Remove the oldest non-system message from the context

        Returns:
            int: Number of tokens freed
        """
        if len(self.messages) <= 1:  # Keep system message
            return 0

        # Remove oldest message (index 1, after system message)
        removed_message = self.messages.pop(1)
        tokens_freed = self.count_message_tokens(removed_message)
        self.total_tokens -= tokens_freed

        if self.track_token_usage:
            logger.debug(f"Pruned message, freed {tokens_freed} tokens")

        return tokens_freed

    def add_message(self, role: str, content: str) -> bool:
        """
        Add a message to the conversation history with token management

        Args:
            role (str): Message role (user/assistant)
            content (str): Message content
        Returns:
            bool: True if message was added successfully
        """
        try:
            # Initialize messages if not present
            if not hasattr(self, 'messages') or self.messages is None:
                logger.warning("Messages list not initialized, creating it now")
                self.messages = []

                # Also add a system message if it's the first initialization
                if len(self.messages) == 0:
                    system_msg = {
                        "role": "system",
                        "content": """You are an expert content writer creating cohesive, engaging articles."""
                    }
                    self.messages.append(system_msg)

            message = {"role": role, "content": content}

            # Initialize tokenizer if not present
            if not hasattr(self, 'encoding') or self.encoding is None:
                logger.warning("Tokenizer not initialized, trying to create it now")
                try:
                    import tiktoken
                    self.encoding = tiktoken.encoding_for_model(self.openai_engine)
                except Exception as e:
                    logger.error(f"Failed to initialize tokenizer: {str(e)}")
                    # Fallback to a basic tokenizer
                    try:
                        import tiktoken
                        self.encoding = tiktoken.get_encoding("cl100k_base")
                    except:
                        # Create a dummy encoding method as last resort
                        class DummyEncoding:
                            def encode(self, text):
                                return [0] * (len(text) // 4)  # Rough approximation
                        self.encoding = DummyEncoding()
                        logger.error("Using emergency dummy tokenizer in add_message")

            # Initialize total_tokens if not present
            if not hasattr(self, 'total_tokens'):
                logger.warning("total_tokens not initialized, setting to 0")
                self.total_tokens = 0

            tokens_needed = self.count_message_tokens(message)

            # First add the new message
            self.messages.append(message)
            self.total_tokens += tokens_needed

            # Then keep pruning oldest messages until we're within limits
            while (
                self.total_tokens > self.max_context_window_tokens - self.token_padding
            ):
                if len(self.messages) <= 1:  # Keep system message
                    # Remove the message we just added since we can't make space
                    self.messages.pop()
                    self.total_tokens -= tokens_needed
                    logger.warning("Cannot free enough tokens even after pruning")
                    return False

                # Remove oldest message (index 1, after system message)
                removed_message = self.messages.pop(1)
                tokens_freed = self.count_message_tokens(removed_message)
                self.total_tokens -= tokens_freed

                if self.track_token_usage:
                    logger.debug(f"Pruned oldest message, freed {tokens_freed} tokens")

            # Warn if approaching token limit
            if (
                self.track_token_usage
                and self.total_tokens
                > self.max_context_window_tokens * self.warn_token_threshold
            ):
                logger.warning(
                    f"Token usage at {(self.total_tokens / self.max_context_window_tokens) * 100:.1f}% of maximum"
                )

            return True

        except Exception as e:
            error_handler.handle_error(e, severity="error")
            return False

    def set_rag_context(self, rag_context: str) -> None:
        """
        Set RAG context in the system message to enhance all subsequent responses
        without explicitly mentioning it in each prompt.
        
        Args:
            rag_context (str): RAG context to add to system message
        """
        if not rag_context:
            return

        try:
            logger.info("Setting RAG context in system message...")
            # Get the current system message
            system_msg = next((msg for msg in self.messages if msg["role"] == "system"), None)
            if not system_msg:
                logger.warning("No system message found to augment with RAG context")
                return

            # Calculate tokens first
            old_tokens = self.count_message_tokens(system_msg)

            # Update system message with RAG context
            enhanced_system_content = f"""
            {system_msg["content"]}

            Additional context information to use for all responses:

            {rag_context}

            Use the above contextual information to inform all your responses without explicitly mentioning that you're using this information.
            """

            # Update the system message in place
            system_msg["content"] = enhanced_system_content.strip()

            # Recalculate tokens
            new_tokens = self.count_message_tokens(system_msg)
            self.total_tokens += (new_tokens - old_tokens)

            logger.success(f"RAG context added to system message ({len(rag_context)} characters)")
            logger.debug(f"Token update: +{new_tokens - old_tokens} tokens")

        except Exception as e:
            error_handler.handle_error(e, severity="error")
            logger.error(f"Error setting RAG context: {str(e)}")

    def get_token_usage_stats(self) -> Dict[str, int]:
        """Get current token usage statistics"""
        try:
            return {
                "total_tokens": self.total_tokens,
                "available_tokens": self.get_available_tokens(),
                "max_tokens": self.max_context_window_tokens,
                "usage_percentage": (self.total_tokens / self.max_context_window_tokens)
                * 100,
            }
        except Exception as e:
            error_handler.handle_error(e, severity="error")
            return {
                "total_tokens": 0,
                "available_tokens": self.max_context_window_tokens,
                "max_tokens": self.max_context_window_tokens,
                "usage_percentage": 0,
            }

    def save_to_file(self, filename: str = None) -> str:
        """
        Save the ArticleContext to a markdown file

        Args:
            filename (str, optional): Filename to save to. If None, a default name will be generated.

        Returns:
            str: Path to the saved file
        """
        if not self.config.enable_context_save:
            return None

        # Create directory if it doesn't exist
        os.makedirs(self.config.context_save_dir, exist_ok=True)

        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_keyword = (
                self.keyword.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]
            )
            filename = f"{safe_keyword}_{timestamp}_context.md"

        filepath = os.path.join(self.config.context_save_dir, filename)

        # Create markdown content
        md_content = f"# Article Context for: {self.keyword}\n\n"
        md_content += (
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        # Add configuration section
        md_content += "## Configuration\n\n"
        md_content += f"- Article Type: {self.articletype}\n"
        md_content += f"- Language: {self.articlelanguage}\n"
        md_content += f"- Voice Tone: {self.voicetone}\n"
        md_content += f"- Point of View: {self.pointofview}\n"
        md_content += f"- Target Audience: {self.articleaudience}\n"
        md_content += f"- OpenAI Engine: {self.openai_engine}\n\n"

        # Add token usage section
        md_content += "## Token Usage\n\n"
        md_content += f"- Total Tokens Used: {self.total_tokens_used}\n"
        md_content += f"- Completion Tokens: {self.completion_tokens}\n"
        md_content += f"- Prompt Tokens: {self.prompt_tokens}\n"
        md_content += f"- Max Context Window: {self.max_context_window_tokens}\n"
        md_content += f"- Available Tokens: {self.get_available_tokens()}\n\n"

        # Add conversation history
        md_content += "## Conversation History\n\n"
        for i, msg in enumerate(self.messages):
            role = msg["role"].capitalize()
            content = msg["content"].replace("\n", "\n> ")
            md_content += f"### Message {i+1} ({role})\n\n> {content}\n\n"

        # Add article parts
        md_content += "## Generated Article Parts\n\n"
        for part_name, part_content in self.article_parts.items():
            if part_content:
                if part_name == "sections":
                    md_content += f"### Sections\n\n"
                    for i, section in enumerate(part_content):
                        md_content += f"#### Section {i+1}\n\n{section}\n\n"
                else:
                    md_content += f"### {part_name.capitalize()}\n\n{part_content}\n\n"

        # Write to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_content)
            logger.success(f"Article context saved to {filepath}")
            return filepath
        except Exception as e:
            error_handler.handle_error(e, severity="error")
            return None

# Define retry decorator for OpenAI API calls
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def make_openai_api_call(messages, model, temperature=0.7, max_tokens=None, seed=None,
                    top_p=None, frequency_penalty=None, presence_penalty=None):
    """
    Make API call to OpenAI with retries.

    Args:
        messages: List of message dictionaries to send to the API
        model: Model ID to use
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        seed: Optional seed for deterministic generation
        top_p: Top-p sampling parameter
        frequency_penalty: Frequency penalty parameter
        presence_penalty: Presence penalty parameter

    Returns:
        OpenAI API response
    """
    try:
        # Build parameters dictionary
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # Add optional parameters if provided
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if seed is not None:
            params["seed"] = seed
        if top_p is not None:
            params["top_p"] = top_p
        if frequency_penalty is not None:
            params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            params["presence_penalty"] = presence_penalty

        # Log the parameters for debugging
        logger.debug(f"OpenAI API parameters: {params}")

        # Make the API call
        response = openai.chat.completions.create(**params)
        return response
    except Exception as e:
        error_handler.handle_error(e, severity="error")
        raise

# New function for OpenRouter API calls with circuit breaker
# @circuit_breaker("openrouter", failure_threshold=3, reset_timeout=60)
@retry(
    stop=stop_after_attempt(5),  # Maximum 5 retries for all errors
    wait=wait_exponential(multiplier=10, min=65, max=120)  # Minimum 65s wait for free tier limits, max 120s
)
def make_openrouter_api_call(messages, model, api_key, site_url, site_name, temperature=0.7, max_tokens=None, seed=None,
                      top_p=None, frequency_penalty=None, presence_penalty=None):
    """
    Make API call to OpenRouter with retries.

    Args:
        messages: List of message dictionaries to send to the API
        model: Model ID to use
        api_key: OpenRouter API key
        site_url: Website URL for attribution
        site_name: Website name for attribution
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        seed: Optional seed for deterministic generation
        top_p: Top-p sampling parameter
        frequency_penalty: Frequency penalty parameter
        presence_penalty: Presence penalty parameter

    Returns:
        OpenRouterResponse: Response object with a format similar to OpenAI's
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": site_url or "https://example.com",
            "X-Title": site_name or "AI Article Generator"
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        # ADD the order parameter to payload specifically for model names containing the word gemini
        if "gemini" in model.lower():
            # For Gemini models, prioritize the 'Google AI Studio' provider
            payload["provider"] = {
            "order": ["Google AI Studio"]
            }

        # Add optional parameters if provided
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        if top_p is not None:
            payload["top_p"] = top_p
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty

        # Log the request payload for debugging
        logger.debug(f"OpenRouter API request payload: {json.dumps(payload, indent=2)}")

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )

        # Specific handling for rate limit errors (HTTP 429)
        if response.status_code == 429:
            rate_limit_info = response.text
            logger.warning(f"OpenRouter rate limit exceeded: {rate_limit_info}")
            
            # Extract rate limit details if available
            try:
                error_data = response.json()
                error_message = error_data.get("error", {}).get("message", "Unknown rate limit error")
                logger.warning(f"Rate limit details: {error_message}")
                
                # Check if it's the free tier rate limit
                if "free-models-per-min" in error_message:
                    logger.warning("Free tier rate limit hit. Consider upgrading to paid tier or using a different model.")
                    # Raise a custom error for minute-based rate limit
                    raise RateLimitError(
                        f"OpenRouter free tier rate limit: {error_message}", 
                        is_minute_limit=True, 
                        retry_after=65  # Wait slightly longer than a minute
                    )
            except (ValueError, KeyError):
                pass
                
            # Raise a generic rate limit error
            raise RateLimitError(f"OpenRouter rate limit exceeded: {rate_limit_info}", retry_after=20)

        elif response.status_code != 200:
            logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
            response.raise_for_status()

        result = response.json()
        logger.debug(f"OpenRouter API response: {json.dumps(result, indent=2)}")

        # Check if the result contains an error field
        if "error" in result:
            error_msg = result.get("error", {}).get("message", "Unknown error")
            logger.error(f"OpenRouter API returned error: {error_msg}")
            
            # Handle rate limit errors in the response body
            if "rate limit" in error_msg.lower() or "rate_limit" in error_msg.lower():
                logger.warning(f"Rate limit error detected in response: {error_msg}")
                
                # Check if it's the free tier minute-based rate limit
                is_minute_limit = "free-models-per-min" in error_msg.lower()
                retry_after = 65 if is_minute_limit else 10
                
                raise RateLimitError(
                    f"OpenRouter rate limit in response: {error_msg}",
                    is_minute_limit=is_minute_limit,
                    retry_after=retry_after
                )
            
            # For other errors, raise a generic exception
            raise ValueError(f"OpenRouter API error: {error_msg}")

        # Check if the response has the expected structure
        if "choices" not in result:
            error_msg = f"Unexpected OpenRouter API response format: {result}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Handle different response formats from OpenRouter
        # Some models might return different structures
        choices = result["choices"]
        if not choices:
            error_msg = "Empty choices in OpenRouter API response"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get the first choice
        choice = choices[0]

        # Check if the choice has the expected structure
        if "message" not in choice:
            error_msg = f"Unexpected choice format in OpenRouter API response: {choice}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        message = choice["message"]

        # Check if the message has content
        if "content" not in message:
            error_msg = f"No content in message from OpenRouter API response: {message}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        content = message["content"]

        # Get usage information if available
        usage = result.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})

        # Convert to an object similar to OpenAI's response format
        class OpenRouterResponse:
            class Choice:
                class Message:
                    def __init__(self, content):
                        self.content = content
                def __init__(self, message_content):
                    self.message = self.Message(message_content)
            class Usage:
                def __init__(self, prompt_tokens, completion_tokens):
                    self.prompt_tokens = prompt_tokens
                    self.completion_tokens = completion_tokens
                    self.total_tokens = prompt_tokens + completion_tokens

                def get(self, key, default=0):
                    """Dictionary-like get method for compatibility with article_context.update_token_usage"""
                    if key == 'prompt_tokens':
                        return self.prompt_tokens
                    elif key == 'completion_tokens':
                        return self.completion_tokens
                    elif key == 'total_tokens':
                        return self.total_tokens
                    return default
            def __init__(self, choices, usage):
                self.choices = [self.Choice(choices[0]["message"]["content"])]
                self.usage = self.Usage(usage["prompt_tokens"], usage["completion_tokens"])

        return OpenRouterResponse(
            choices,
            usage
        )
    except Exception as e:
        logger.error(f"OpenRouter API call failed: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise

def gpt_completion(
    context: ArticleContext,
    prompt: str,
    temp: Optional[float] = None,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    generation_type: str = "content_generation",
    allowed: str = "controlled"
) -> str:
    """
    Generate text completion using either OpenAI or OpenRouter API with dynamic context management.

    Args:
        context: Article context object
        prompt: The prompt to send to the API
        temp: Temperature for generation (if None, will use the appropriate temperature from config based on generation_type)
        max_tokens: Maximum tokens to generate (if None, will use the appropriate max_tokens from config based on generation_type)
        seed: Optional seed for deterministic generation
        generation_type: Type of content being generated (used to select appropriate temperature and max_tokens from config)

    Returns:
        Generated text
    """
    # Bismillahir rahmanir raheem - In the name of Allah, the Most Gracious, the Most Merciful

    # Initialize parameters with default values
    top_p = None
    frequency_penalty = None
    presence_penalty = None

    # If temperature is not provided, use the appropriate one from config based on generation_type
    if temp is None:
        if generation_type == "content_generation":
            temp = getattr(context.config, "content_generation_temperature", 1.0)
            top_p = getattr(context.config, "content_generation_top_p", None)
            frequency_penalty = getattr(context.config, "content_generation_frequency_penalty", None)
            presence_penalty = getattr(context.config, "content_generation_presence_penalty", None)
            logger.debug(f"Using content_generation_temperature: {temp}")
        elif generation_type == "title":
            temp = getattr(context.config, "content_generation_temperature", 1.0)
            top_p = getattr(context.config, "content_generation_top_p", None)
            frequency_penalty = getattr(context.config, "content_generation_frequency_penalty", None)
            presence_penalty = getattr(context.config, "content_generation_presence_penalty", None)
            logger.debug(f"Using content_generation_temperature for title: {temp}")
        elif generation_type == "outline":
            temp = getattr(context.config, "content_generation_temperature", 1.0)
            top_p = getattr(context.config, "content_generation_top_p", None)
            frequency_penalty = getattr(context.config, "content_generation_frequency_penalty", None)
            presence_penalty = getattr(context.config, "content_generation_presence_penalty", None)
            logger.debug(f"Using content_generation_temperature for outline: {temp}")
        elif generation_type == "meta_description":
            temp = getattr(context.config, "meta_description_temperature", 0.7)
            top_p = getattr(context.config, "meta_description_top_p", None)
            frequency_penalty = getattr(context.config, "meta_description_frequency_penalty", None)
            presence_penalty = getattr(context.config, "meta_description_presence_penalty", None)
            logger.debug(f"Using meta_description_temperature: {temp}")
        elif generation_type == "humanization":
            temp = getattr(context.config, "humanization_temperature", 0.7)
            top_p = getattr(context.config, "humanization_top_p", None)
            frequency_penalty = getattr(context.config, "humanization_frequency_penalty", None)
            presence_penalty = getattr(context.config, "humanization_presence_penalty", None)
            logger.debug(f"Using humanization_temperature: {temp}")
        elif generation_type == "grammar_check":
            temp = getattr(context.config, "grammar_check_temperature", 0.3)
            top_p = getattr(context.config, "grammar_check_top_p", None)
            frequency_penalty = getattr(context.config, "grammar_check_frequency_penalty", None)
            presence_penalty = getattr(context.config, "grammar_check_presence_penalty", None)
            logger.debug(f"Using grammar_check_temperature: {temp}")
        elif generation_type == "block_notes":
            temp = getattr(context.config, "block_notes_temperature", 1.0)
            top_p = getattr(context.config, "block_notes_top_p", None)
            frequency_penalty = getattr(context.config, "block_notes_frequency_penalty", None)
            presence_penalty = getattr(context.config, "block_notes_presence_penalty", None)
            logger.debug(f"Using block_notes_temperature: {temp}")
        elif generation_type == "faq_generation":
            temp = getattr(context.config, "faq_generation_temperature", 1.0)
            top_p = getattr(context.config, "faq_generation_top_p", None)
            frequency_penalty = getattr(context.config, "faq_generation_frequency_penalty", None)
            presence_penalty = getattr(context.config, "faq_generation_presence_penalty", None)
            logger.debug(f"Using faq_generation_temperature: {temp}")
        else:
            # Default to content_generation_temperature for unknown types
            temp = getattr(context.config, "content_generation_temperature", 1.0)
            top_p = getattr(context.config, "content_generation_top_p", None)
            frequency_penalty = getattr(context.config, "content_generation_frequency_penalty", None)
            presence_penalty = getattr(context.config, "content_generation_presence_penalty", None)
            logger.debug(f"Using default content_generation_temperature for {generation_type}: {temp}")
    else:
        logger.debug(f"Using provided temperature: {temp} for {generation_type}")

    # If max_tokens is not provided, use the appropriate one from config based on generation_type
    if max_tokens is None:
        if generation_type == "content_generation":
            max_tokens = getattr(context.config, "content_generation_max_tokens", 600)
            logger.debug(f"Using content_generation_max_tokens: {max_tokens}")
        elif generation_type == "title":
            max_tokens = getattr(context.config, "title_max_tokens", 100)
            logger.debug(f"Using title_max_tokens: {max_tokens}")
        elif generation_type == "outline":
            max_tokens = getattr(context.config, "outline_max_tokens", 500)
            logger.debug(f"Using outline_max_tokens: {max_tokens}")
        elif generation_type == "introduction":
            max_tokens = getattr(context.config, "introduction_max_tokens", 600)
            logger.debug(f"Using introduction_max_tokens: {max_tokens}")
        elif generation_type == "paragraph":
            max_tokens = getattr(context.config, "paragraph_max_tokens", 600)
            logger.debug(f"Using paragraph_max_tokens: {max_tokens}")
        elif generation_type == "conclusion":
            max_tokens = getattr(context.config, "conclusion_max_tokens", 600)
            logger.debug(f"Using conclusion_max_tokens: {max_tokens}")
        elif generation_type == "meta_description":
            max_tokens = getattr(context.config, "meta_description_max_tokens", 200)
            logger.debug(f"Using meta_description_max_tokens: {max_tokens}")
        elif generation_type == "humanization":
            max_tokens = getattr(context.config, "humanization_max_tokens", 600)
            logger.debug(f"Using humanization_max_tokens: {max_tokens}")
        elif generation_type == "grammar_check":
            max_tokens = getattr(context.config, "grammar_check_max_tokens", 600)
            logger.debug(f"Using grammar_check_max_tokens: {max_tokens}")
        elif generation_type == "block_notes":
            max_tokens = getattr(context.config, "block_notes_max_tokens", 300)
            logger.debug(f"Using block_notes_max_tokens: {max_tokens}")
        elif generation_type == "faq_generation":
            max_tokens = getattr(context.config, "faq_generation_max_tokens", 1000)
            logger.debug(f"Using faq_generation_max_tokens: {max_tokens}")
        else:
            # Default to content_generation_max_tokens for unknown types
            max_tokens = getattr(context.config, "content_generation_max_tokens", 600)
            logger.debug(f"Using default content_generation_max_tokens for {generation_type}: {max_tokens}")
    else:
        logger.debug(f"Using provided max_tokens: {max_tokens} for {generation_type}")

    # Create format kwargs dict with all possible formatting parameters
    format_kwargs = {
        "keyword": context.keyword,
        "articletype": context.articletype,
        "articlelanguage": context.articlelanguage,
        "voicetone": context.voicetone,
        "pointofview": context.pointofview,
        "articleaudience": context.articleaudience,
        "size_headings": context.size_headings,
        "size_sections": context.size_sections,
        "sizeheadings": context.size_headings,  # For backward compatibility
        "sizesections": context.size_sections,  # For backward compatibility
    }

    # Only try to format if the prompt contains formatting placeholders
    if "{" in prompt and "}" in prompt:
        try:
            formatted_prompt = prompt.format(**format_kwargs)
        except KeyError as e:
            logger.warning(f"Missing format key: {e}, using original prompt")
            formatted_prompt = prompt
    else:
        formatted_prompt = prompt

    try:
        
        # Add user message
        context.add_message("user", formatted_prompt)

        # Prepare messages for API call
        if not hasattr(context, 'messages') or not context.messages:
            logger.warning("Messages attribute missing or empty in context, initializing it")
            # Initialize with system message if missing
            system_msg = {
                "role": "system",
                "content": """You are an expert content writer creating cohesive, engaging articles in RAW HTML."""
            }
            context.messages = [system_msg]
            context.add_message("user", formatted_prompt)

        messages = context.messages

        # Attempt to make the API call with retries
        try:
            # Check if using OpenRouter or direct OpenAI
            if context.config.use_openrouter and context.config.openrouter_api_key:
                logger.info(f"Using OpenRouter with model: {context.openrouter_model}")

                # Use the configured OpenRouter model
                model_to_use = context.openrouter_model

                # If model is in our list of OpenRouter models, use the full path
                if context.config.openrouter_models:
                    for key, full_model_id in context.config.openrouter_models.items():
                        if key.lower() in model_to_use.lower():
                            model_to_use = full_model_id
                            logger.info(f"Mapped to OpenRouter model: {model_to_use}")
                            break

                # FIXED: No longer removing previous user messages to preserve full context
                # Instead, implement smarter token-aware context pruning if needed
                if hasattr(context, 'messages') and len(context.messages) > 3:
                    # Check if we're approaching token limits
                    total_tokens = sum(context.count_message_tokens(msg) for msg in messages)
                    max_allowed = context.max_context_window_tokens - context.token_padding
                    
                    if total_tokens > max_allowed:
                        logger.warning(f"Context size ({total_tokens} tokens) exceeds limit ({max_allowed}), pruning older messages")
                        # Keep system message, most recent assistant message, and current user message
                        # First, separate messages by role
                        system_msgs = [msg for msg in messages if msg["role"] == "system"]
                        user_msgs = [msg for msg in messages if msg["role"] == "user"]
                        assistant_msgs = [msg for msg in messages if msg["role"] == "assistant"]
                        
                        # Keep all system messages, the last few assistant messages, and only the current user message
                        preserved_msgs = system_msgs + assistant_msgs[-2:] + [user_msgs[-1]]
                        messages = preserved_msgs
                        logger.info(f"Pruned context to {len(messages)} messages")

                # Get additional parameters based on generation_type
                top_p = None
                frequency_penalty = None
                presence_penalty = None

                if generation_type == "content_generation":
                    top_p = getattr(context.config, "content_generation_top_p", 1.0)
                    frequency_penalty = getattr(context.config, "content_generation_frequency_penalty", 0.0)
                    presence_penalty = getattr(context.config, "content_generation_presence_penalty", 0.0)
                elif generation_type == "meta_description":
                    top_p = getattr(context.config, "meta_description_top_p", 1.0)
                    frequency_penalty = getattr(context.config, "meta_description_frequency_penalty", 0.0)
                    presence_penalty = getattr(context.config, "meta_description_presence_penalty", 0.0)
                elif generation_type == "humanization":
                    top_p = getattr(context.config, "humanization_top_p", 1.0)
                    frequency_penalty = getattr(context.config, "humanization_frequency_penalty", 0.0)
                    presence_penalty = getattr(context.config, "humanization_presence_penalty", 0.0)
                elif generation_type == "grammar_check":
                    top_p = getattr(context.config, "grammar_check_top_p", 1.0)
                    frequency_penalty = getattr(context.config, "grammar_check_frequency_penalty", 0.0)
                    presence_penalty = getattr(context.config, "grammar_check_presence_penalty", 0.0)
                elif generation_type == "block_notes":
                    top_p = getattr(context.config, "block_notes_top_p", 1.0)
                    frequency_penalty = getattr(context.config, "block_notes_frequency_penalty", 0.0)
                    presence_penalty = getattr(context.config, "block_notes_presence_penalty", 0.0)
                elif generation_type == "faq_generation":
                    top_p = getattr(context.config, "faq_generation_top_p", 1.0)
                    frequency_penalty = getattr(context.config, "faq_generation_frequency_penalty", 0.0)
                    presence_penalty = getattr(context.config, "faq_generation_presence_penalty", 0.0)

                # Log the parameters being used
                logger.debug(f"Using parameters for {generation_type}: temp={temp}, top_p={top_p}, " +
                           f"frequency_penalty={frequency_penalty}, presence_penalty={presence_penalty}")

                response = make_openrouter_api_call(
                    messages=messages,
                    model=model_to_use,
                    api_key=context.config.openrouter_api_key,
                    site_url=context.config.openrouter_site_url,
                    site_name=context.config.openrouter_site_name,
                    temperature=temp,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
                
                
            else:
                # Use direct OpenAI API with the same parameters as OpenRouter
                response = make_openai_api_call(
                    messages=messages,
                    model=context.openai_engine,
                    temperature=temp,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
                

            # Get the generated text
            generated_text = response.choices[0].message.content.strip()

            # Add assistant response to context
            context.add_message("assistant", generated_text)

            # Update token usage stats
            if context.track_token_usage and hasattr(response, "usage"):
                context.update_token_usage(response.usage)

            time.sleep(3)

            # Extract content from XML tags based on prompt analysis
            def extract_content_from_xml_tags(response_text: str, prompt: str) -> str:
                """
                Extract content from XML tags based on tags mentioned in the prompt.
                If no tags are found or extraction fails, return the original response.
                """
                import re
                
                # Find XML tags mentioned in the prompt
                xml_tag_pattern = r'<([a-zA-Z][a-zA-Z0-9_]*)>'
                mentioned_tags = re.findall(xml_tag_pattern, prompt)
                
                if not mentioned_tags:
                    # No XML tags found in prompt, return original response
                    return response_text
                
                # Get the most likely tag (usually the last one mentioned)
                target_tag = mentioned_tags[-1]
                
                # Create regex pattern to extract content from the target tag
                content_pattern = fr'<{target_tag}[^>]*>(.*?)</{target_tag}>'
                
                # Search for the content within the target tag
                match = re.search(content_pattern, response_text, re.DOTALL | re.IGNORECASE)
                
                if match:
                    # Successfully extracted content from the tag
                    extracted_content = match.group(1).strip()
                    logger.debug(f"Extracted content from <{target_tag}> tags")
                    return extracted_content
                else:
                    # Tag not found in response, return original response as fallback
                    logger.warning(f"Expected <{target_tag}> tags not found in response, using full response")
                    return response_text

            # Apply XML tag extraction
            generated_text = extract_content_from_xml_tags(generated_text, formatted_prompt)

            # Define allowed tags
            if allowed == "controlled":
                allowed_tags = ('em', 'strong', 'paragraph', 'heading')
            else:
                allowed_tags = ('em', 'strong', 'paragraph', 'heading', "ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td")

            # Function to check if a tag is allowed; if not, remove it
            def remove_unwanted_tags(match):
                tag = match.group(0)
                # Get the tag name without angle brackets or slashes
                tag_name = re.sub(r'[</>]', '', tag).split()[0].lower()
                if tag_name in allowed_tags:
                    return tag  # Keep allowed tags
                return ""  # Remove all other tags

            # Regex pattern to match any HTML tag
            html_tag_pattern = r'</?[a-zA-Z][^>]*>'

            # Remove all HTML tags except allowed ones while preserving their inner text
            generated_text = re.sub(html_tag_pattern, remove_unwanted_tags, generated_text)

            # Remove code blocks and clean up
            generated_text = generated_text.replace('```html', '').replace('```', '').strip()

            return generated_text
        except RateLimitError as e:
            # Handle rate limit errors
            logger.warning(f"Rate limit error in API call during {generation_type}: {str(e)}")
            
            # Handle free-models-per-min rate limit specifically
            if hasattr(e, 'is_minute_limit') and e.is_minute_limit:
                wait_time = e.retry_after if hasattr(e, 'retry_after') else 65
                logger.warning(f"Free tier minute-based rate limit hit. Waiting for {wait_time} seconds before retry...")
                time.sleep(wait_time)
                
                # Retry the request after waiting
                logger.info("Retrying after rate limit wait period...")
                return gpt_completion(
                    context=context,
                    prompt=prompt,
                    temp=temp,
                    max_tokens=max_tokens,
                    seed=seed,
                    generation_type=generation_type,
                    allowed= allowed
                )
            
            # For other rate limits, wait a shorter time or re-raise
            wait_time = e.retry_after if hasattr(e, 'retry_after') else 10
            logger.warning(f"Standard rate limit hit. Waiting for {wait_time} seconds before retry...")
            time.sleep(wait_time)
            
            # Retry the request after waiting
            logger.info("Retrying after rate limit wait period...")
            return gpt_completion(
                context=context,
                prompt=prompt,
                temp=temp,
                max_tokens=max_tokens,
                seed=seed,
                generation_type=generation_type,
                allowed= allowed
            )
        except Exception as e:
            logger.error(f"Error in API call during {generation_type}: {str(e)}")
            # Emergency fallback - generate some basic text
            fallback_text = f"Information about {context.keyword} related to {generation_type}."
            logger.warning(f"Using fallback text: {fallback_text}")
            return fallback_text
    except Exception as e:
        logger.error(f"Error in gpt_completion preparing request: {str(e)}")
        # Emergency fallback for critical errors
        return f"Information about {context.keyword}."

def generate_title(context: ArticleContext, keyword: str, prompt_template: str) -> str:
    """
    Generate a title for the article.

    Args:
        context: Article context object
        keyword: Main keyword for the article
        prompt_template: Template for title generation

    Returns:
        Generated title
    """
    logger.info(f"Generating title for keyword: {keyword}")

    # Format the prompt
    prompt = prompt_template.format(
        keyword=keyword,
        voicetone=context.voicetone,
        articletype=context.articletype,
        articlelanguage=context.articlelanguage,
        articleaudience=context.articleaudience,
    )

    try:
        # Generate title - use the config value for max_tokens
        title = gpt_completion(
            context,
            prompt,
            generation_type="title",
            seed=context.config.title_seed if context.config.enable_seed_control else None,
        )

        # Store title in context
        context.article_parts["title"] = title

        logger.success(f"Generated title: {title}")
        return title
    except Exception as e:
        logger.error(f"Error generating title: {str(e)}")
        default_title = f"Article about {keyword}"
        context.article_parts["title"] = default_title
        return default_title

def generate_outline(
    context: ArticleContext, keyword: str, prompt_template: str
) -> str:
    """
    Generate an outline for the article.

    Args:
        context: Article context object
        keyword: Main keyword for the article
        prompt_template: Template for outline generation

    Returns:
        Generated outline
    """
    logger.info(f"Generating outline for: {keyword}")

    # Create a dynamic example outline based on the configuration
    example_outline = ""
    roman_numerals = ["I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "X."]
    subsection_letters = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J."]
    
    for i in range(min(context.size_sections, len(roman_numerals))):
        example_outline += f"{roman_numerals[i]} [Main Section Title]\n"
        
        for j in range(min(context.size_headings, len(subsection_letters))):
            example_outline += f"{subsection_letters[j]} [Subsection Point]\n"
        
        # Add blank line between sections
        if i < context.size_sections - 1:
            example_outline += "\n"

    # Format the prompt
    prompt = prompt_template.format(
        keyword=keyword,
        size_headings=context.size_headings,
        size_sections=context.size_sections,
        sizesections=context.config.sizesections,
        sizeheadings=context.config.sizeheadings,
        sizeparagraphs=context.config.paragraphs_per_section,
        voicetone=context.voicetone,
        articletype=context.articletype,
        articlelanguage=context.articlelanguage,
        articleaudience=context.articleaudience,
        pointofview=context.pointofview,
        example_outline=example_outline
    )

    # Generate outline - use the config value for max_tokens
    outline = gpt_completion(
        context,
        prompt,
        generation_type="outline",
        seed=context.config.outline_seed if context.config.enable_seed_control else None
    )

    # Clean up the outline
    outline = outline.replace("**", "").strip()

    # Store outline in context
    context.article_parts["outline"] = outline

    logger.success(f"Generated outline with {outline.count('#')} sections")
    return outline

def generate_introduction(
    context: ArticleContext, keyword: str, prompt_template: str
) -> str:
    """Generate article introduction."""
    logger.info(f"Generating introduction for keyword: {keyword}")

    prompt = prompt_template.format(
        keyword=keyword,
        articletype=context.articletype,
        articlelanguage=context.articlelanguage,
        voicetone=context.voicetone,
        pointofview=context.pointofview,
        articleaudience=context.articleaudience,
    )

    # Use introduction seed if seed control is enabled
    seed = (
        context.config.introduction_seed if context.config.enable_seed_control else None
    )

    introduction = gpt_completion(
        context=context,
        prompt=prompt,
        generation_type="introduction",  # Use introduction-specific settings
        seed=seed
    )
    context.article_parts["introduction"] = introduction

    logger.success("\n" + "=" * 50)
    logger.success("Generated Introduction:")
    logger.success("-" * 50)
    for paragraph in introduction.split("\n\n"):
        if paragraph.strip():
            logger.success(paragraph.strip() + "\n")
    logger.success("=" * 50 + "\n")

    return introduction

def parse_outline(outline: str, context: ArticleContext = None) -> List[Dict[str, any]]:
    """
    Parse the outline into a structured format based on strict formatting rules.
    Supports both 2-level (sections → subsections) and 3-level (sections → subsections → paragraph points) hierarchies.

    Args:
        outline (str): The generated outline that follows the strict format:
            I. Main Section
            A. Subsection
               1. Paragraph point
               2. Paragraph point
            B. Subsection
               1. Paragraph point

            II. Main Section
            ...
        context (ArticleContext, optional): Article context with configuration.
            
    Returns:
        List[Dict[str, any]]: List of sections with their details and hierarchical structure
    """
    logger.info("Starting outline parsing...")

    # Split into lines and clean up
    lines = [line.strip() for line in outline.split("\n") if line.strip()]

    logger.info(f"Found {len(lines)} non-empty lines")

    sections = []
    current_section = None
    current_subsection = None
    
    # Use context's size_headings directly if provided, otherwise fallback to 5
    expected_subsections = context.size_headings if context and hasattr(context, 'size_headings') else 5

    # Define valid markers
    roman_numerals = ["I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "X.", "XI.", "XII.", "XIII.", "XIV.", "XV."]
    subsection_letters = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J.", "K.", "L.", "M.", "N.", "O."]
    paragraph_numbers = ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.", "11.", "12.", "13.", "14.", "15."]

    # Track position in the outline
    current_main_section = 0
    current_subsection_index = 0
    current_paragraph_index = 0
    
    # Check if we should expect subsections based on size_headings
    expect_subsections = expected_subsections > 0

    for line in lines:
        # Check for main section
        if any(line.lstrip().startswith(numeral) for numeral in roman_numerals):
            current_main_section += 1
            current_subsection_index = 0
            current_paragraph_index = 0

            # Extract section title
            for numeral in roman_numerals:
                if line.lstrip().startswith(numeral):
                    title = line[line.find(numeral) + len(numeral) :].strip()
                    logger.info(
                        f"Processing main section {current_main_section}: '{title}'"
                    )

                    if not title:
                        logger.error("Empty main section title")
                        continue

                    current_section = {
                        "title": title,
                        "subsections": [],
                        "has_paragraph_points": False  # Flag to indicate 3-level structure
                    }
                    sections.append(current_section)
                    break

        # Check for subsection (only if we expect subsections)
        elif expect_subsections and current_section and any(
            line.lstrip().startswith(letter) for letter in subsection_letters
        ):
            current_subsection_index += 1
            current_paragraph_index = 0

            # Extract subsection content
            for letter in subsection_letters:
                if line.lstrip().startswith(letter):
                    content = line[line.find(letter) + len(letter) :].strip()
                    logger.debug(
                        f"Processing subsection {current_subsection_index}: '{content}'"
                    )

                    if not content:
                        logger.error("Empty subsection content")
                        continue

                    current_subsection = {
                        "title": content,
                        "paragraph_points": []
                    }
                    current_section["subsections"].append(current_subsection)
                    break

        # Check for paragraph points (2-level hierarchy when no subsections)
        elif not expect_subsections and current_section and any(
            line.lstrip().startswith(number) for number in paragraph_numbers
        ):
            current_paragraph_index += 1

            # Extract paragraph point content as direct subsection
            for number in paragraph_numbers:
                if line.lstrip().startswith(number):
                    content = line[line.find(number) + len(number) :].strip()
                    logger.debug(
                        f"Processing direct paragraph point {current_paragraph_index}: '{content}'"
                    )

                    if not content:
                        logger.error("Empty paragraph point content")
                        continue

                    # Create a subsection directly from paragraph point
                    subsection = {
                        "title": content,
                        "paragraph_points": [content]  # Use the same content as paragraph point
                    }
                    current_section["subsections"].append(subsection)
                    break

        # Check for paragraph points (3-level hierarchy with subsections)
        elif expect_subsections and current_section and current_section["subsections"] and any(
            line.lstrip().startswith(number) for number in paragraph_numbers
        ):
            current_paragraph_index += 1

            # Extract paragraph point content
            for number in paragraph_numbers:
                if line.lstrip().startswith(number):
                    content = line[line.find(number) + len(number) :].strip()
                    logger.debug(
                        f"Processing paragraph point {current_paragraph_index}: '{content}'"
                    )

                    if not content:
                        logger.error("Empty paragraph point content")
                        continue

                    # Add to the last subsection
                    if current_section["subsections"]:
                        current_section["subsections"][-1]["paragraph_points"].append(content)
                        current_section["has_paragraph_points"] = True
                    break

        else:
            # Skip empty or unrecognized lines without warning
            if line.strip():
                logger.debug(f"Skipping line: '{line}'")

    # Validate the parsed outline
    if not sections:
        logger.error("No valid sections found in outline")
        return []

    # Determine structure type and validate
    has_3_level = any(section.get("has_paragraph_points", False) for section in sections)
    
    # Final debug output
    logger.success("Parsing complete!")
    logger.success(f"Found {len(sections)} main sections")
    
    for section in sections:
        subsection_count = len(section["subsections"])
        logger.success(
            f"Section '{section['title']}' has {subsection_count} subsections"
        )
        
        # If 3-level structure, log paragraph points
        if has_3_level:
            for i, subsection in enumerate(section["subsections"], 1):
                paragraph_count = len(subsection.get("paragraph_points", []))
                logger.success(
                    f"  Subsection {i}: '{subsection['title']}' has {paragraph_count} paragraph points"
                )

    return sections

def generate_section(
    context: ArticleContext,
    heading: str,
    keyword: str,
    section_number: int,
    total_sections: int,
    paragraph_prompt: str,
    parsed_sections: List[Dict[str, any]],
) -> str:
    """
    Generate content for a section of the article, handling both 2-level and 3-level hierarchical structures.

    Args:
        context: Article context object
        heading: Section heading
        keyword: Main keyword for the article
        section_number: Current section number
        total_sections: Total number of sections
        paragraph_prompt: Template for paragraph generation
        parsed_sections: List of parsed section dictionaries with hierarchical structure

    Returns:
        Generated section content with proper HTML structure
    """
    logger.info(f"Generating section {section_number}/{total_sections}: {heading}")

    # Find the current section data
    current_section = None
    for section in parsed_sections:
        if section["title"] == heading:
            current_section = section
            break
    
    if not current_section and len(parsed_sections) >= section_number:
        # Fallback: get section by index if title match fails
        current_section = parsed_sections[section_number - 1]

    # Determine hierarchy type and extract appropriate content
    has_paragraph_points = current_section.get("has_paragraph_points", False)
    subsections = current_section.get("subsections", [])
    
    logger.info(f"Section '{heading}' has {len(subsections)} subsections")
    logger.info(f"Hierarchy type: {'3-level (with paragraph points)' if has_paragraph_points else '2-level (subsections only)'}")

    # Generate content based on hierarchy type
    if has_paragraph_points:
        # 3-level hierarchy: sections → subsections → paragraph points
        return _generate_3level_section(
            context, heading, keyword, section_number, total_sections,
            subsections, has_paragraph_points
        )
    else:
        # 2-level hierarchy: sections → subsections
        return _generate_2level_section(
            context, heading, keyword, section_number, total_sections,
            subsections, paragraph_prompt, parsed_sections
        )
    # HEADING_GUIDELLINES=f"""
    # • NO HTML tags or any other formatting
    # • NO special characters or symbols
    # • NO emojis or informal language
    # • NO abbreviations or acronyms unless commonly known
    # • NO repetition of words or phrases
    # • NO TALKING, AS THIS WILL BE USED FOR AN API
    # • NO explanations or comments
    # • NO extra text or context
    # • NO unnecessary details or filler content
    # • NO comments such as here is your heading, or here is your outline etc.
    # • NO unnecessary words or phrases
    # • NO filler content or irrelevant information
    # • NO personal opinions or subjective statements
    # • NO assumptions about the reader's knowledge or background
    # • NO references to external sources or citations
    # • NO use of ** or any kind of markdown etc.
    # • NO quotes or paraphrasing from other sources
    # • NO use of "blog" or similar terms
    # • NO use of "article" or similar terms
    # • NO use of "content" or similar terms

    # REFERENCE HEADING: {heading}
    # """
    # message_to_generate_a_good_heading = f"""Generate a good heading for the following content:-
    # {section_content}
    # MAKE SURE TO FOLLOW THE FOLLOWING guidelines:-
    # {HEADING_GUIDELLINES}
    # """
    # # Generate a new heading using OpenRouter API
    # new_heading_using_open_router = make_openrouter_api_call(
    #     messages=[
    #         {"role": "user", "content": message_to_generate_a_good_heading}
    #     ],
    #     api_key=context.config.openrouter_api_key,
    #     temperature=0.7,
    #     max_tokens=100,
    #     model=context.openrouter_model,
    #     site_name=context.config.openrouter_site_name,
    #     site_url=context.config.openrouter_site_url,
    # )
    # error_handler.handle_error(Exception(new_heading_using_open_router)

    # new_heading_using_open_router = new_heading_using_open_router.choices[0].message.content.strip()

    # formatted_section = f"## {heading}\n\n{section_content}"

    # # Store section in context
    # context.article_parts["sections"].append(formatted_section)

    # logger.success(f"Generated section {section_number}: {len(section_content)} chars")
    # return formatted_section

def generate_conclusion(
    context: ArticleContext, keyword: str, conclusion_prompt: str, summarize_prompt: str
) -> str:
    """Generate article conclusion."""
    logger.info(f"Generating conclusion for keyword: {keyword}")

    prompt = conclusion_prompt.format(
        keyword=keyword,
        articletype=context.articletype,
        articlelanguage=context.articlelanguage,
        voicetone=context.voicetone,
        pointofview=context.pointofview,
        articleaudience=context.articleaudience,
    )

    # Use conclusion seed if seed control is enabled
    seed = (
        context.config.conclusion_seed if context.config.enable_seed_control else None
    )

    conclusion = gpt_completion(
        context=context,
        prompt=prompt,
        generation_type="conclusion",  # Use conclusion-specific settings
        seed=seed
    )
    context.article_parts["conclusion"] = conclusion

    logger.success("\n" + "=" * 50)
    logger.success("Generated Conclusion:")
    logger.success("-" * 50)
    for paragraph in conclusion.split("\n\n"):
        if paragraph.strip():
            logger.success(paragraph.strip() + "\n")
    logger.success("=" * 50 + "\n")

    return conclusion

def generate_complete_article(
    context: ArticleContext,
    keyword: str,
    prompts: Prompts,
    image_keyword: Optional[str] = None,
    sections: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Generate a complete article with context awareness.
    """
    logger.info(f"\n{'='*20} Starting Article Generation {'='*20}")
    logger.info(f"Keyword: {keyword}")
    logger.info(f"Article Type: {context.articletype}")
    logger.info(f"Language: {context.articlelanguage}")
    logger.info(f"Voice Tone: {context.voicetone}")
    logger.info(f"Point of View: {context.pointofview}")
    logger.info(f"Target Audience: {context.articleaudience}")
    logger.info("=" * 60 + "\n")

    # Generate title using the prompt from the prompts configuration
    title = generate_title(context, keyword, prompts.title)

    # Generate outline
    outline = generate_outline(context, keyword, prompts.outline)

    # Parse the outline to get main sections
    parsed_sections = parse_outline(outline)

    # Generate introduction using the prompt from configuration
    introduction = generate_introduction(context, keyword, prompts.introduction)

    # Generate main content
    content_sections = []
    total_sections = len(parsed_sections)

    for i, section in enumerate(parsed_sections, 1):
        content = generate_section(
            context=context,
            heading=section["title"],
            keyword=keyword,
            section_number=i,
            total_sections=total_sections,
            paragraph_prompt=prompts.paragraph_generate,
            parsed_sections=parsed_sections,
        )
        content_sections.append(content)

    # Generate conclusion using both conclusion and summary prompts
    conclusion = generate_conclusion(
        context=context,
        keyword=keyword,
        conclusion_prompt=prompts.conclusion,
        summarize_prompt=prompts.summarize,
    )

    # Log completion and token usage
    logger.success(f"\n{'='*20} Article Generation Complete {'='*20}")
    if context.track_token_usage:
        stats = context.get_token_usage_stats()
        logger.info("Token Usage Statistics:")
        logger.info(f"Total Tokens Used: {stats['total_tokens']}")
        logger.info(f"Available Tokens: {stats['available_tokens']}")
        logger.info(f"Usage Percentage: {stats['usage_percentage']:.1f}%")
    logger.success("=" * 60 + "\n")

    # Return complete article
    return {
        "title": title,
        "outline": outline,
        "introduction": introduction,
        "sections": content_sections,
        "conclusion": conclusion,
    }

def generate_article(context: ArticleContext) -> str:
    """Generate the complete article content."""
    logger.info("Starting article generation...")

    # Generate outline first
    outline = generate_outline(
        context=context,
        keyword=context.keyword,
        prompt_template=context.config.outline_prompt if hasattr(context.config, 'outline_prompt') else context.prompts.outline
    )
    parsed_sections = parse_outline(outline)

    # Generate introduction
    introduction = generate_introduction(
        context=context,
        keyword=context.keyword,
        prompt_template=context.config.introduction_prompt if hasattr(context.config, 'introduction_prompt') else context.prompts.introduction
    )

    # Generate each section
    content_sections = []
    for i, section in enumerate(parsed_sections, 1):
        content = generate_section(
            context=context,
            heading=section["title"],  # Using 'title' key
            keyword=context.keyword,
            section_number=i,
            total_sections=len(parsed_sections),
            paragraph_prompt=context.config.paragraph_generate_prompt if hasattr(context.config, 'paragraph_generate_prompt') else context.prompts.paragraph_generate,
            parsed_sections=parsed_sections
        )
        content_sections.append(content)

    # Generate conclusion
    conclusion = generate_conclusion(
        context=context,
        keyword=context.keyword,
        conclusion_prompt=context.config.conclusion_prompt if hasattr(context.config, 'conclusion_prompt') else context.prompts.conclusion,
        summarize_prompt=context.config.summarize_prompt if hasattr(context.config, 'summarize_prompt') else context.prompts.summarize
    )

    # Combine all parts
    article_parts = [introduction] + content_sections + [conclusion]
    complete_article = "\n\n".join(article_parts)

    return complete_article

def generate_article_summary(
    context: ArticleContext,
    keyword: str,
    article_dict: Dict[str, str],
    summarize_prompt: str,
    combine_prompt: str,
) -> str:
    """
    Generate a comprehensive summary of the article.

    Uses a large context window model if configured, with chunking for very large articles.
    """
    logger.info("Generating article summary...")

    try:
        # Determine if we should use a separate model for summary generation
        use_separate_model = (
            hasattr(context.config, 'enable_separate_summary_model') and
            context.config.enable_separate_summary_model and
            hasattr(context.config, 'summary_keynotes_model') and
            context.config.summary_keynotes_model
        )

        # Get the chunk size from config or use default
        chunk_size = getattr(context.config, 'summary_chunk_size', 8000)

        # Import chunking utilities
        from article_generator.chunking_utils import chunk_article_for_processing, combine_chunk_results, combine_chunk_results_with_llm

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
            )
            # Add main content with proper formatting
            full_content += "Main Content:\n{0}\n\n".format('\n'.join(chunk.get('sections', [])))
            # Add conclusion
            full_content += f"Conclusion: {chunk.get('conclusion', '')}"

            # Format the summary prompt
            prompt = summarize_prompt.format(
                keyword=keyword,
                articleaudience=context.articleaudience,
                article_content=full_content,
            )

            # Use key_takeaways seed if seed control is enabled
            seed = (
                context.config.key_takeaways_seed
                if context.config.enable_seed_control
                else None
            )

            # Generate summary with the appropriate model
            if use_separate_model and context.config.use_openrouter:
                logger.info(f"Using separate model for summary generation: {context.config.summary_keynotes_model}")

                # Use OpenRouter with the specified model
                from article_generator.content_generator import make_openrouter_api_call

                # Get max tokens from config or use default
                max_tokens = getattr(context.config, 'summary_max_tokens', 800)

                # Create messages for the API call
                messages = [
                    {"role": "system", "content": "You are an expert content writer specializing in creating comprehensive article summaries."},
                    {"role": "user", "content": prompt}
                ]

                # Make the API call
                # Get temperature and other parameters from config or use defaults
                temperature = getattr(context.config, "content_generation_temperature", 1.0)
                summary_top_p = getattr(context.config, "content_generation_top_p", None)
                summary_frequency_penalty = getattr(context.config, "content_generation_frequency_penalty", None)
                summary_presence_penalty = getattr(context.config, "content_generation_presence_penalty", None)

                response = make_openrouter_api_call(
                    messages=messages,
                    model=context.config.summary_keynotes_model,
                    api_key=context.config.openrouter_api_key,
                    site_url=context.config.openrouter_site_url,
                    site_name=context.config.openrouter_site_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=summary_top_p,
                    frequency_penalty=summary_frequency_penalty,
                    presence_penalty=summary_presence_penalty
                )

                chunk_summary = response.choices[0].message.content.strip()
            else:
                # Use the standard gpt_completion function
                chunk_summary = gpt_completion(
                    context,
                    prompt,
                    max_tokens=getattr(context.config, 'summary_max_tokens', 800),
                    generation_type="content_generation",
                    seed=seed,
                )

            if chunk_summary:
                words = chunk_summary.split()
                if len(words) > 220:
                    chunk_summary = " ".join(words[:200]) + "..."
                chunk_results.append(chunk_summary)
            else:
                logger.warning("No summary was generated from chunk")
                return ""

        # Combine results from all chunks
        if not chunk_results:
            logger.warning("No summary was generated from any chunk")
            return ""

        # Use the LLM to combine chunks if there are multiple chunks
        if len(chunk_results) > 1:
            logger.info("Using LLM to combine summary chunks")
            summary = combine_chunk_results_with_llm(chunk_results, context, combine_prompt, is_summary=True)
        else:
            summary = chunk_results[0]

        logger.success(f"Generated article summary ({len(summary.split())} words)")
        return summary.strip()

    except Exception as e:
        logger.error(f"Error generating article summary: {str(e)}")
        # Return empty string on error rather than raising
        return ""

from bs4 import BeautifulSoup

# Flow instruction logic for paragraph positioning, supporting all possible HTML output formats with optional structure
def get_flow_instruction(current_paragraph, paragraphs_per_section):
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
     
def generate_paragraph(
    context: ArticleContext,
    heading: str,
    keyword: str,
    lsi_keywords: List[str] = None,
    current_paragraph: int = 1,
    paragraphs_per_section: int = None,
    section_number: int = 1,
    total_sections: int = 1,
    section_points: List[str] = None,
    web_context: str = ""
) -> str:
    """Generate paragraph content in a single API call, optimized for SEO and readability."""
    logger.debug(f"Generating paragraph {current_paragraph}/{paragraphs_per_section} for: {heading}")

    # Define allowed HTML tags for all parsing paths
    allowed_tags = ["strong", "em", "ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td"]

    # Set defaults if not provided
    if paragraphs_per_section is None:
        paragraphs_per_section = context.paragraphs_per_section
    
    if section_points is None:
        section_points = []
    
    if lsi_keywords is None:
        lsi_keywords = []

    # Format all points and LSI keywords as strings
    all_points_str = "\n".join([f"- {point}" for point in section_points])
    lsi_keywords_str = ", ".join(lsi_keywords) if lsi_keywords else "none provided"

    # Distribute points across paragraphs
    current_points = []
    if section_points and len(section_points) > 0:
        points_per_paragraph = max(1, len(section_points) // paragraphs_per_section)
        start_idx = (current_paragraph - 1) * points_per_paragraph
        end_idx = min(start_idx + points_per_paragraph, len(section_points))
        current_points = section_points[start_idx:end_idx]
    
    current_points_str = "\n".join([f"- {point}" for point in current_points]) if current_points else "- General information about this topic"

    # Adjust flow instruction based on position in section
    # Flow instruction logic for paragraph positioning
    flow_instruction = get_flow_instruction(current_paragraph,paragraphs_per_section)
  

    # Create format kwargs
    format_kwargs = {
        "primary_keyword": keyword,
        "lsi_keywords": lsi_keywords_str,
        "heading": heading,
        "section_number": section_number,
        "total_sections": total_sections,
        "paragraphs_per_section": paragraphs_per_section,
        "current_paragraph": current_paragraph,
        "voicetone": context.voicetone,
        "articletype": context.articletype,
        "articlelanguage": context.articlelanguage,
        "articleaudience": context.articleaudience,
        "pointofview": context.pointofview,
        "all_points": all_points_str,
        "current_points": current_points_str,
        "flow_instruction": flow_instruction
    }

    # Format the prompt
    try:
        from prompts_simplified import PARAGRAPH_PROMPT
        prompt = PARAGRAPH_PROMPT.format(**format_kwargs)
    except Exception as e:
        logger.error(f"Error formatting paragraph prompt: {e}")
        prompt = f"Write paragraph {current_paragraph} of {paragraphs_per_section} for section '{heading}' about {keyword}. Use HTML tags <strong>, <em>, <ul><li>, <ol><li>, or <table> with <thead>, <tr>, <th>, <td> as appropriate, and include LSI keywords: {lsi_keywords_str}. Return in format: <paragraph>Paragraph content</paragraph><heading><h3>Heading text</h3></heading>"

    # Add outline context for reference
    outline_context = "\n".join([
        f"# Article Title: {context.article_parts['title']}",
        f"# Article Outline:",
    ] + [f"Section {j+1}: {section['title']}" for j, section in enumerate(context.article_parts.get('parsed_outline', []))])
    
    prompt = f"{prompt}\n\n{outline_context}"
    logger.debug(f"Formatted prompt: {prompt}")

    # Use paragraph seed if seed control is enabled
    seed = context.config.paragraph_seed if context.config.enable_seed_control else None

    # Generate the paragraph with heading
    try:
        response = gpt_completion(
            context=context,
            prompt=prompt,
            generation_type="paragraph",
            temp=context.config.content_generation_temperature,
            max_tokens=context.config.paragraph_max_tokens,
            seed=seed,
            allowed= "all"
        )
        
        logger.debug(f"Response from GPT: {response}")

        formatted_paragraph = f'<p>{response}</p>'
        return formatted_paragraph

    except Exception as e:
        logger.error(f"Error generating paragraph with heading: {str(e)}")
        # Return a fallback paragraph with heading in case of error
        return f'<h4>About {keyword}</h4>\n\n<p>Information about {heading} related to {keyword}.</p>'

def _generate_2level_section(
    context: ArticleContext,
    heading: str,
    keyword: str,
    section_number: int,
    total_sections: int,
    subsections: List[Dict],
    paragraph_prompt: str,
    parsed_sections: List[Dict]
) -> str:
    """
    Generate content for a 2-level hierarchy (sections → subsections).
    
    Args:
        context: Article context object
        heading: Section heading
        keyword: Main keyword for the article
        section_number: Current section number
        total_sections: Total number of sections
        paragraph_prompt: Template for paragraph generation
        parsed_sections: List of parsed section dictionaries
        subsections: List of subsection dictionaries
        
    Returns:
        Generated section content
    """
    logger.info(f"Generating 2-level section: {heading}")
    
    # Format subsection points as strings
    subsection_titles = [sub["title"] for sub in subsections]
    all_points_str = "\n".join([f"- {title}" for title in subsection_titles])
    
    # Calculate tokens per paragraph
    tokens_per_paragraph = context.section_token_limit // context.paragraphs_per_section
    
    # Use paragraph seed if seed control is enabled
    seed = context.config.paragraph_seed if context.config.enable_seed_control else None
    
    # Generate multiple paragraphs
    paragraphs = []
    
    for i in range(context.paragraphs_per_section):
        current_paragraph = i + 1
        logger.debug(f"Generating paragraph {current_paragraph}/{context.paragraphs_per_section}")
        
        if context.config.enable_paragraph_headings:
            # Generate paragraph with heading in a single API call
            formatted_paragraph = generate_paragraph(
                context=context,
                heading=heading,
                keyword=keyword,
                current_paragraph=current_paragraph,
                paragraphs_per_section=context.paragraphs_per_section,
                section_number=section_number,
                total_sections=total_sections,
                section_points=subsection_titles
            )
            paragraphs.append(formatted_paragraph)
            continue
        
        # Create format kwargs for this specific paragraph
        format_kwargs = {
            "keyword": keyword,
            "heading": heading,
            "section_number": section_number,
            "total_sections": total_sections,
            "paragraphs_per_section": context.paragraphs_per_section,
            "current_paragraph": i + 1,
            "min_paragraph_tokens": context.min_paragraph_tokens,
            "max_paragraph_tokens": context.max_paragraph_tokens,
            "voicetone": context.voicetone,
            "articletype": context.articletype,
            "articlelanguage": context.articlelanguage,
            "articleaudience": context.articleaudience,
            "pointofview": context.pointofview,
            "all_points": all_points_str,
            "total_points": len(subsection_titles),
        }

        # Format the prompt
        try:
            prompt = paragraph_prompt.format(**format_kwargs)
        except KeyError as e:
            logger.warning(f"Missing key in paragraph prompt: {e}, using simplified prompt")
            guidelines = """
            Make sure to not include any markdown formatting as this will be used to create an article using html.
            Never ever wrap the text in **, or something similar. Give the section a good name and make sure to wrap it in a h2 heading, and also wrap the subsections in h3 headings and so on.
            Make sure to only return valid HTML, which could directly be used inside wordpress as a post.
            Make sure to not include any markdown formatting as this will be used to create an article using html.
            Make sure to use proper formatting tags such as em, strong etc. as per the rules of modern HTML.
            Make sure to style the content with proper HTML formatting, and don't return any errors, as this will directly be used for a wordpress post.
            Never ever return ```, or  `, or ```html etc. or things such as here is your code etc.
            """
            prompt = f"Write paragraph {i+1} of {context.paragraphs_per_section} for section '{heading}' about {keyword}. But follow the following guidelines: {guidelines}"

        # Add outline context for reference
        outline_context = "\n".join([
            f"# Article Title: {context.article_parts['title']}",
            f"# Article Outline:",
        ] + [f"Section {j+1}: {section['title']}" for j, section in enumerate(parsed_sections)])

        prompt = f"{prompt}\n\n{outline_context}"

        # Generate this paragraph
        try:
            paragraph = gpt_completion(
                context=context,
                prompt=prompt,
                generation_type="paragraph",
                seed=seed,
                allowed="all"
            )
            formatted_paragraph = f'<p>{paragraph}</p>'
            paragraphs.append(formatted_paragraph)
        except Exception as e:
            logger.error(f"Error generating paragraph {i+1} for '{heading}': {str(e)}")
            fallback_paragraph = f'<p>Information about {heading} related to {keyword}.</p>'
            paragraphs.append(fallback_paragraph)

    # Join all paragraphs with double newlines
    section_content = "\n\n".join(paragraphs)
    formatted_section = f"## {heading}\n\n{section_content}"

    # Store section in context
    context.article_parts["sections"].append(formatted_section)

    logger.success(f"Generated 2-level section {section_number}: {len(section_content)} chars")
    return formatted_section

def _generate_3level_section(
    context: ArticleContext,
    heading: str,
    keyword: str,
    section_number: int,
    total_sections: int,
    subsections: List[Dict],
    has_paragraph_points: bool
) -> str:
    """
    Generate content for a 3-level hierarchy (sections → subsections → paragraph points).
    
    Args:
        context: Article context object
        heading: Section heading
        keyword: Main keyword for the article
        section_number: Current section number
        total_sections: Total number of sections
        subsections: List of subsection dictionaries with paragraph points
        
    Returns:
        Generated section content with subsections
    """
    logger.info(f"Generating 3-level section: {heading}")
    
    section_content = f"## {heading}\n\n"
    
    # Process each subsection
    for subsection in subsections:
        subsection_title = subsection["title"]
        paragraph_points = subsection.get("paragraph_points", [])
        
        # Add subsection heading
        section_content += f"### {subsection_title}\n\n"
        
        # Generate content for each paragraph point
        for point in paragraph_points:
            try:
                # Create prompt for this specific point
                prompt = f"""Write a paragraph about: {point}
                
                Context: This is part of section "{heading}" about {keyword}.
                Target audience: {context.articleaudience}
                Tone: {context.voicetone}
                Language: {context.articlelanguage}
                
                Return only the paragraph content in HTML format."""
                
                paragraph = gpt_completion(
                    context=context,
                    prompt=prompt,
                    generation_type="paragraph",
                    allowed="all"
                )
                section_content += f'<p>{paragraph}</p>\n\n'
                
            except Exception as e:
                logger.error(f"Error generating paragraph for point '{point}': {str(e)}")
                section_content += f'<p>{point}</p>\n\n'
    
    # Store section in context
    context.article_parts["sections"].append(section_content)

    logger.success(f"Generated 3-level section {section_number}: {len(section_content)} chars")
    return section_content