# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import openai
from typing import Optional, Dict, Any
import time
import json
import requests
from utils.rate_limiter import openai_rate_limiter
from tenacity import retry, stop_after_attempt, wait_exponential
from article_generator.logger import logger
import re

# Custom exception for rate limit errors
class RateLimitError(Exception):
    """Exception raised when API rate limits are hit."""
    
    def __init__(self, message, is_minute_limit=False, retry_after=None):
        """Initialize the exception.
        
        Args:
            message: Error message
            is_minute_limit: Whether this is a minute-based rate limit (e.g., free-models-per-min)
            retry_after: Suggested time to wait before retrying (in seconds)
        """
        super().__init__(message)
        self.is_minute_limit = is_minute_limit
        self.retry_after = retry_after or 60  # Default to 60 seconds for free tier

@retry(
    stop=stop_after_attempt(5),  # Maximum 5 retries for all errors
    wait=wait_exponential(multiplier=10, min=65, max=120)  # Minimum 65s wait for free tier limits, max 120s
)
def make_openrouter_api_call(
    messages: list,
    model: str,
    api_key: str,
    site_url: str,
    site_name: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0
):
    """Make API call to OpenRouter with enhanced error handling for rate limits.

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
        logger.info(f"Making OpenRouter API call with model: {model}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": site_url,
            "X-Title": site_name
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty
        }

        if "gemini" in model:
            time.sleep(3)
            payload["provider"] = {
            "order": ["Google AI Studio"]
            }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if seed is not None:
            payload["seed"] = seed

        # Log the request payload for debugging
        logger.debug(f"OpenRouter API request payload: {json.dumps(payload, indent=2)}")
        
        time.sleep(10)
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
                
            # Raise the HTTP error for retry handling
            response.raise_for_status()

        elif response.status_code != 200:
            logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
            response.raise_for_status()

        result = response.json()
        logger.debug(f"OpenRouter API response: {result}")

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
                
            # For other errors, wait and retry
            time.sleep(10)
            logger.info("Retrying OpenRouter API call...")

            # Log the retry request payload for debugging
            logger.debug(f"OpenRouter API retry request payload: {json.dumps(payload, indent=2)}")

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            if response.status_code != 200:
                logger.error(f"OpenRouter API error: {response.status_code} - {response.text}")
                response.raise_for_status()
            result = response.json()

        # Check if the response has the expected structure
        if "choices" not in result:
            error_msg = f"Unexpected OpenRouter API response format: {result}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Handle different response formats from OpenRouter
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
        raise

def generate_completion(
    prompt: str,
    model: str,
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    article_context: Optional[Any] = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: Optional[int] = None,
    allowed: str = "controlled"
) -> str:
    """
    Generate text using OpenAI's GPT model or OpenRouter API with enhanced rate limit handling.
    Args:
        prompt: Prompt to send to the model
        model: Model to use (OpenAI model or key in openrouter_models)
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        article_context: Context object for the article
        top_p: Top-p sampling parameter
        frequency_penalty: Frequency penalty parameter
        presence_penalty: Presence penalty parameter
        seed: Optional seed for deterministic generation
        allowed: Tag filtering mode ("controlled" or "all")
    Returns:
        str: Generated text
    """
    try:
        # Get config from article context
        config = None
        if article_context and hasattr(article_context, 'config'):
            config = article_context.config
        # Ensure API key is set for OpenAI
        if not openai.api_key and config:
            openai.api_key = config.openai_key
        # Prepare messages - Use ArticleContext message history if available
        messages = []
        # If we have an ArticleContext with messages, use those
        if article_context and hasattr(article_context, 'messages') and article_context.messages:
            logger.info("Using existing message history from ArticleContext")
            # Log the number of messages being used
            logger.debug(f"Using {len(article_context.messages)} messages from context history")
            
            # Add the user prompt to the context
            # article_context.add_message("user", prompt)
            
            # Use the full message history for the API call
            messages = article_context.messages
        else:
            # Fallback if no ArticleContext is provided or it has no messages
            logger.info("No ArticleContext available, creating new message array")
            
            # Add system message if available
            if config and hasattr(config, 'prompts'):
                system_message = config.prompts.system_message
                if system_message:
                    messages.append({"role": "system", "content": system_message})
            
            # Add user prompt
            messages.append({"role": "user", "content": prompt})
            
            # Add the message to ArticleContext if available (even if it didn't have messages before)
            if article_context and hasattr(article_context, 'add_message'):
                article_context.add_message("user", prompt)
        # Determine if we should use OpenRouter
        use_openrouter = False
        if config and hasattr(config, 'use_openrouter'):
            use_openrouter = config.use_openrouter and config.openrouter_api_key
        # Get generated text based on API choice
        if use_openrouter:
            logger.info(f"Using OpenRouter with model: {model}")
            # Prioritize the passed model parameter (which will be the grammar/humanization model when those features are enabled)
            # This fixes the issue where grammar and humanization models weren't being respected when OpenRouter was enabled
            model_to_use = model
            # Set reasonable max_tokens for RAG requests
            if max_tokens is None:
                max_tokens = 4000  # Default for RAG requests
            elif max_tokens > 10000:
                max_tokens = 10000  # Cap at 10k tokens for safety
            # Check if model needs to be mapped to OpenRouter full path
            if config and config.openrouter_models:
                # If model is in our list of OpenRouter models, use the full path
                for key, full_model_id in config.openrouter_models.items():
                    if key.lower() in model_to_use.lower():
                        model_to_use = full_model_id
                        logger.info(f"Mapped to OpenRouter model: {model_to_use}")
                        break
            # Log the parameters being used for OpenRouter
            logger.debug(f"Using parameters for OpenRouter: model={model_to_use}, temperature={temperature}, " +
                     f"max_tokens={max_tokens}, top_p={top_p}, " +
                     f"frequency_penalty={frequency_penalty}, presence_penalty={presence_penalty}, seed={seed}")
            # Define OpenRouter API call
            def make_api_call():
                return make_openrouter_api_call(
                    messages=messages,
                    model=model_to_use,
                    api_key=config.openrouter_api_key,
                    site_url=config.openrouter_site_url or "https://example.com",
                    site_name=config.openrouter_site_name or "AI Article Generator",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty
                )
            # Execute with rate limiting if available
            try:
                if openai_rate_limiter:
                    response = openai_rate_limiter.execute_with_rate_limit(make_api_call)
                else:
                    response = make_api_call()
            except requests.exceptions.HTTPError as e:
                # Check for rate limit errors
                if "429" in str(e) or "rate limit" in str(e).lower():
                    logger.warning(f"OpenRouter rate limit error: {str(e)}")
                    
                    # Check if it's the free tier minute-based rate limit
                    is_minute_limit = "free-models-per-min" in str(e).lower()
                    retry_after = 65 if is_minute_limit else 10
                    
                    # Let the caller handle this with their retry mechanism
                    raise RateLimitError(
                        f"OpenRouter rate limit: {str(e)}", 
                        is_minute_limit=is_minute_limit,
                        retry_after=retry_after
                    )
                else:
                    # Re-raise other HTTP errors
                    raise
        else:
            # Use OpenAI API directly
            # Prepare API parameters
            api_params = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }
            # Add seed if provided
            if seed is not None:
                api_params["seed"] = seed
            # Log the API parameters for debugging
            logger.debug(f"OpenAI API parameters: {json.dumps(api_params, indent=2, default=str)}")
            # Execute API call with rate limiting if available
            if openai_rate_limiter:
                logger.debug("Using rate limiter for OpenAI API call")
                def make_api_call():
                    return openai.chat.completions.create(**api_params)
                response = openai_rate_limiter.execute_with_rate_limit(make_api_call)
            else:
                response = openai.chat.completions.create(**api_params)
        
        # Extract generated text
        generated_text = response.choices[0].message.content.strip()
        
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
        generated_text = extract_content_from_xml_tags(generated_text, prompt)
        
        if allowed == "all":
             # Define allowed tags
             allowed_tags = ('em', 'strong', 'paragraph', 'heading',"ul", "ol", "li", "table", "thead", "tbody", "tr", "th", "td")        
        else:
             # Define allowed tags
             allowed_tags = ('em', 'strong', 'paragraph', 'heading')
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
        # Remove all HTML tags except allowed ones while preserving their inner text.
        generated_text = re.sub(html_tag_pattern, remove_unwanted_tags, generated_text)
        # Add assistant response to context if context is provided
        if article_context and hasattr(article_context, 'add_message'):
            logger.info("Adding assistant response to context")
            content_preview = generated_text[:50] + "..." if len(generated_text) > 50 else generated_text
            logger.debug(f"Response content: '{content_preview}'")
            # Log the current state of the context before adding
            if hasattr(article_context, 'messages'):
                logger.info(f"Context before adding: {len(article_context.messages)} messages")
            article_context.add_message("assistant", generated_text)
            # Log the current state of the context after adding
            if hasattr(article_context, 'messages'):
                logger.info(f"Context after adding: {len(article_context.messages)} messages")
            # Update token usage stats if available
            if hasattr(article_context, 'update_token_usage') and hasattr(response, 'usage'):
                try:
                    # Handle different usage formats (OpenAI vs OpenRouter)
                    if hasattr(response.usage, 'get'):
                        # OpenRouter format
                        usage = {
                            'prompt_tokens': response.usage.get('prompt_tokens', 0),
                            'completion_tokens': response.usage.get('completion_tokens', 0),
                            'total_tokens': response.usage.get('total_tokens', 0)
                        }
                        article_context.update_token_usage(usage)
                    else:
                        # OpenAI format
                        article_context.update_token_usage(response.usage)
                except Exception as e:
                    logger.error(f"Error updating token usage: {str(e)}")
                    # Continue without updating token usage
        return generated_text
    except openai.RateLimitError as e:
        logger.warning(f"OpenAI rate limit reached: {str(e)}. Waiting before retry...")
        time.sleep(20)
        # Retry with reduced token count if max_tokens is specified
        if max_tokens:
            logger.info(f"Retrying with reduced token count: {max(200, max_tokens // 2)}")
            return generate_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max(200, max_tokens // 2),  # Reduce tokens but keep minimum of 200
                article_context=article_context,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                allowed=allowed
            )
        # Re-raise for the caller to handle if we can't reduce tokens
        raise
        
    except RateLimitError as e:
        logger.warning(f"Rate limit error: {str(e)}")
        
        # Handle free-models-per-min rate limit specifically
        if hasattr(e, 'is_minute_limit') and e.is_minute_limit:
            wait_time = e.retry_after if hasattr(e, 'retry_after') else 65
            logger.warning(f"Free tier minute-based rate limit hit. Waiting for {wait_time} seconds before retry...")
            time.sleep(wait_time)
            
            # Retry the request after waiting
            logger.info("Retrying after rate limit wait period...")
            return generate_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                article_context=article_context,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                seed=seed,
                allowed=allowed
            )
        
        # For other rate limits, wait a shorter time or re-raise
        wait_time = e.retry_after if hasattr(e, 'retry_after') else 10
        logger.warning(f"Standard rate limit hit. Waiting for {wait_time} seconds before retry...")
        time.sleep(wait_time)
        
        # Retry the request after waiting
        logger.info("Retrying after rate limit wait period...")
        return generate_completion(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            article_context=article_context,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            allowed=allowed
        )
        
    except Exception as e:
        logger.error(f"Error in generate_completion: {str(e)}")
        return "Error generating response. Please try again."