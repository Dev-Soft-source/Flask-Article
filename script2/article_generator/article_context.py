# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
import tiktoken
import os
import traceback
from datetime import datetime
from utils.rich_provider import provider
from config import Config
from utils.prompts_config import Prompts

@dataclass
class ArticleContext:
    """Maintains conversation context for article generation with token management."""

    config: Config
    prompts: Prompts
    messages: List[Dict[str, str]] = field(default_factory=list)
    article_parts: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = field(default=0)
    encoding: object = field(init=False)

    # Token tracking state
    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens_used: int = 0

    def update_token_usage(self, usage: Dict[str, int]) -> None:
        """Update token usage statistics."""
        if not self.config.enable_token_tracking:
            return

        self.completion_tokens += usage.get('completion_tokens', 0)
        self.prompt_tokens += usage.get('prompt_tokens', 0)
        self.total_tokens_used = self.completion_tokens + self.prompt_tokens

        # Check if we're approaching the context window limit
        if self.total_tokens_used >= (self.config.max_context_window_tokens * self.config.warn_token_threshold):
            provider.warning(f"Approaching token limit. Used {self.total_tokens_used} tokens "
                         f"({(self.total_tokens_used/self.config.max_context_window_tokens)*100:.1f}% of max)")

    def __post_init__(self):
        """Initialize the context with system message and token counter."""
        provider.info("Initializing article context...")

        # Initialize tokenizer - handle OpenRouter models by mapping to compatible OpenAI models
        model_for_tokenizer = self.config.openai_model
        if self.config.use_openrouter:
            # For OpenRouter models, use a compatible tokenizer
            # Most models can use cl100k_base encoding (used by GPT-4 and ChatGPT)
            try:
                provider.info(f"Using tiktoken encoding for OpenRouter model: {model_for_tokenizer}")
                self.encoding = tiktoken.encoding_for_model(model_for_tokenizer)
            except KeyError:
                provider.warning(f"No specific tokenizer for {model_for_tokenizer}, using cl100k_base encoding")
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            # Regular OpenAI model
            self.encoding = tiktoken.encoding_for_model(model_for_tokenizer)

        # Initialize with system message
        system_msg = {
            "role": "system",
            "content": self.prompts.system_message
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
            "block_notes": None,
            "summary": None,
            "faq": None,
            "paa": None,
            "meta_description": None,
            "wordpress_excerpt": None
        }

        if self.config.enable_token_tracking:
            provider.debug(f"Initial token usage: {self.total_tokens}")
            if self.total_tokens > self.config.warn_token_threshold * self.config.max_context_window_tokens:
                provider.warning(f"Token usage is high: {self.total_tokens}/{self.config.max_context_window_tokens}")

        provider.success("Article context initialized successfully")

    def count_message_tokens(self, message: Dict[str, str]) -> int:
        """Count tokens in a message."""
        try:
            return len(self.encoding.encode(message["content"])) + 4  # 4 tokens for message format
        except Exception as e:
            provider.error(f"Error counting tokens: {str(e)}")
            return 0

    def get_available_tokens(self) -> int:
        """Get number of tokens available in the context window."""
        return self.config.max_context_window_tokens - self.total_tokens - self.config.token_padding

    def would_exceed_limit(self, new_content: str) -> Tuple[bool, int]:
        """Check if adding new content would exceed token limits."""
        try:
            tokens_needed = len(self.encoding.encode(new_content))

            if tokens_needed > self.config.openai_token:
                return True, tokens_needed

            if self.total_tokens + tokens_needed > self.config.max_context_window_tokens - self.config.token_padding:
                return True, tokens_needed

            return False, tokens_needed

        except Exception as e:
            provider.error(f"Error checking token limit: {str(e)}")
            return True, 0

    def add_message(self, role: str, content: str) -> bool:
        """Add a message to the context with token management."""
        try:
            message = {"role": role, "content": content}
            tokens_needed = self.count_message_tokens(message)

            # Log message being added (truncate long content for readability)
            content_preview = content[:50] + "..." if len(content) > 50 else content
            provider.info(f"CONTEXT: Adding {role} message to context: '{content_preview}'")
            provider.info(f"CONTEXT: Before adding: {len(self.messages)} messages in context")

            # Log all current message roles for debugging
            message_roles = [msg["role"] for msg in self.messages]
            provider.debug(f"CONTEXT: Current message roles: {message_roles}")

            # First add the new message
            self.messages.append(message)
            self.total_tokens += tokens_needed

            provider.info(f"CONTEXT: After adding: {len(self.messages)} messages in context")

            # Then keep pruning oldest messages until we're within limits
            while self.total_tokens > self.config.max_context_window_tokens - self.config.token_padding:
                if len(self.messages) <= 1:  # Keep system message
                    # Remove the message we just added since we can't make space
                    self.messages.pop()
                    self.total_tokens -= tokens_needed
                    provider.warning("Cannot free enough tokens even after pruning")
                    return False

                # Remove oldest message (index 1, after system message)
                removed_message = self.messages.pop(1)
                tokens_freed = self.count_message_tokens(removed_message)
                self.total_tokens -= tokens_freed

                removed_content = removed_message["content"][:30] + "..." if len(removed_message["content"]) > 30 else removed_message["content"]
                provider.info(f"CONTEXT: Pruned {removed_message['role']} message: '{removed_content}'")

                if self.config.enable_token_tracking:
                    provider.debug(f"Pruned oldest message, freed {tokens_freed} tokens")

            # Warn if approaching token limit
            if self.config.enable_token_tracking and self.total_tokens > self.config.max_context_window_tokens * self.config.warn_token_threshold:
                provider.warning(f"Token usage at {(self.total_tokens / self.config.max_context_window_tokens) * 100:.1f}% of maximum")

            return True

        except Exception as e:
            provider.error(f"Error adding message: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return False

    def get_context_summary(self) -> str:
        """Get a summary of the article content so far."""
        parts = []

        if self.article_parts["title"]:
            parts.append(f"Title: {self.article_parts['title']}")

        if self.article_parts["outline"]:
            parts.append(f"Outline: {self.article_parts['outline']}")

        if self.article_parts["introduction"]:
            parts.append(f"Introduction: {self.article_parts['introduction']}")

        if self.article_parts["sections"]:
            parts.append("\nPrevious sections:")
            for i, section in enumerate(self.article_parts["sections"], 1):
                parts.append(f"Section {i}: {section}")

        return "\n\n".join(parts)

    def clear_messages(self) -> None:
        """Clear message history while preserving system message."""
        system_message = next((msg for msg in self.messages if msg["role"] == "system"), None)
        self.messages = [system_message] if system_message else []
        self.total_tokens = self.count_message_tokens(system_message) if system_message else 0

    def get_token_usage_stats(self) -> Dict[str, Any]:
        """Get current token usage statistics"""
        try:
            return {
                "total_tokens": self.total_tokens,
                "available_tokens": self.get_available_tokens(),
                "max_tokens": self.config.max_context_window_tokens,
                "usage_percentage": (self.total_tokens / self.config.max_context_window_tokens) * 100,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens_used": self.total_tokens_used
            }
        except Exception as e:
            provider.error(f"Error getting token stats: {str(e)}")
            return {
                "total_tokens": 0,
                "available_tokens": self.config.max_context_window_tokens,
                "max_tokens": self.config.max_context_window_tokens,
                "usage_percentage": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens_used": 0
            }

    def get_conversation_context(self) -> List[Dict[str, str]]:
        """Get the current conversation context."""
        return self.messages

    def update_token_stats(self, usage) -> None:
        """Update token statistics from OpenAI response usage."""
        if not hasattr(usage, 'prompt_tokens') or not hasattr(usage, 'completion_tokens'):
            provider.warning("Invalid usage object provided")
            return

        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens_used = self.prompt_tokens + self.completion_tokens

        # Check if approaching limits
        if self.config.enable_token_tracking:
            token_percentage = (self.total_tokens_used / self.config.max_context_window_tokens) * 100
            if token_percentage > self.config.warn_token_threshold * 100:
                provider.warning(f"Token usage at {token_percentage:.1f}% ({self.total_tokens_used}/{self.config.max_context_window_tokens})")

            provider.token_usage(f"Tokens: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total")

    def save_to_file(self, filename: str = None) -> Optional[str]:
        """
        Save the ArticleContext to a markdown file

        Args:
            filename (str, optional): Filename to save to. If None, a default name will be generated.

        Returns:
            str: Path to the saved file or None if saving is disabled or fails
        """
        if not self.config.enable_context_save:
            provider.warning("Context saving is disabled in configuration")
            return None

        # Log the current state of the context
        provider.info(f"CONTEXT SAVE: Saving context with {len(self.messages)} messages")
        message_roles = [msg["role"] for msg in self.messages]
        provider.info(f"CONTEXT SAVE: Message roles: {message_roles}")

        # Create directory if it doesn't exist
        os.makedirs(self.config.context_save_dir, exist_ok=True)

        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Get keyword from article parts if available
            keyword = "article"
            if self.article_parts.get("title"):
                keyword = self.article_parts["title"].replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]
            filename = f"{keyword}_{timestamp}_context.md"

        filepath = os.path.join(self.config.context_save_dir, filename)
        provider.info(f"CONTEXT SAVE: Saving to file: {filepath}")

        # Create markdown content
        md_content = f"# Article Context\n\n"
        md_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Add configuration section
        md_content += "## Configuration\n\n"
        md_content += f"- Article Type: {self.config.articletype}\n"
        md_content += f"- Language: {self.config.articlelanguage}\n"
        md_content += f"- Voice Tone: {self.config.voicetone}\n"
        md_content += f"- Point of View: {self.config.pointofview}\n"
        md_content += f"- Target Audience: {self.config.articleaudience}\n"

        # Add model information
        if self.config.use_openrouter:
            md_content += f"- Model: {self.config.openrouter_model} (via OpenRouter)\n"
        else:
            md_content += f"- Model: {self.config.openai_model}\n"

        # Add token usage statistics
        md_content += "\n## Token Usage\n\n"
        stats = self.get_token_usage_stats()
        md_content += f"- Total Tokens Used: {stats['total_tokens_used']}\n"
        md_content += f"- Prompt Tokens: {stats['prompt_tokens']}\n"
        md_content += f"- Completion Tokens: {stats['completion_tokens']}\n"
        md_content += f"- Context Window Usage: {stats['usage_percentage']:.1f}%\n"

        # Add conversation context
        md_content += "\n## Conversation Context\n\n"
        provider.info(f"CONTEXT SAVE: Writing {len(self.messages)} messages to file")
        for i, msg in enumerate(self.messages):
            if msg["role"] == "system":
                md_content += f"### System Message\n\n{msg['content']}\n\n"
                provider.debug(f"CONTEXT SAVE: Added system message to file")
            else:
                content_preview = msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
                provider.debug(f"CONTEXT SAVE: Adding {msg['role']} message {i}: '{content_preview}'")
                md_content += f"### {msg['role'].capitalize()} Message {i}\n\n{msg['content']}\n\n"

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
            provider.success(f"Article context saved to {filepath}")
            return filepath
        except Exception as e:
            provider.error(f"Error saving article context: {str(e)}")
            provider.error(f"Stack trace:\n{traceback.format_exc()}")
            return None

    def set_rag_context(self, rag_context: str) -> None:
        """
        Set RAG context in the system message to enhance all subsequent responses
        without explicitly mentioning it in each prompt.
        """
        if not rag_context:
            return

        try:

            # save the context to context.txt file
            # with open("context.txt", "w") as f:
            #     f.write(rag_context)
            #     provider.success("RAG context saved to context.txt")
            provider.info("Setting RAG context in system message...")
            # Get the current system message
            system_msg = next((msg for msg in self.messages if msg["role"] == "system"), None)
            if not system_msg:
                provider.warning("No system message found to augment with RAG context")
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

            provider.success(f"RAG context added to system message ({len(rag_context)} characters)")
            provider.debug(f"Token update: +{new_tokens - old_tokens} tokens")

        except Exception as e:
            provider.error(f"Error setting RAG context: {str(e)}")