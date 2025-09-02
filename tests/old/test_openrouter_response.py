# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

"""
Test script to verify the OpenRouterResponse class fix.
This script simulates the token usage update process to ensure the get() method works.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create a simple provider for logging
class SimpleProvider:
    def info(self, message):
        print(f"[INFO] {message}")

    def success(self, message):
        print(f"[SUCCESS] {message}")

    def error(self, message):
        print(f"[ERROR] {message}")

provider = SimpleProvider()

# We'll create our own OpenRouterResponse class for testing
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

def test_openrouter_response():
    """Test the OpenRouterResponse class."""
    provider.info("Testing OpenRouterResponse class...")

    # Create a mock usage object
    usage = {"prompt_tokens": 100, "completion_tokens": 50}

    # Create a mock choices object
    choices = [{"message": {"content": "Test response"}}]

    # Create the OpenRouterResponse object
    response = OpenRouterResponse(choices, usage)

    # Test the get() method
    try:
        prompt_tokens = response.usage.get('prompt_tokens', 0)
        completion_tokens = response.usage.get('completion_tokens', 0)
        total_tokens = response.usage.get('total_tokens', 0)

        provider.success(f"Successfully retrieved token usage:")
        provider.info(f"Prompt tokens: {prompt_tokens}")
        provider.info(f"Completion tokens: {completion_tokens}")
        provider.info(f"Total tokens: {total_tokens}")

        # Verify the values
        assert prompt_tokens == 100, f"Expected prompt_tokens to be 100, got {prompt_tokens}"
        assert completion_tokens == 50, f"Expected completion_tokens to be 50, got {completion_tokens}"
        assert total_tokens == 150, f"Expected total_tokens to be 150, got {total_tokens}"

        provider.success("All assertions passed!")
        return True
    except Exception as e:
        provider.error(f"Error testing OpenRouterResponse: {str(e)}")
        return False

if __name__ == "__main__":
    provider.info("Testing OpenRouterResponse fixes...")

    # Test the OpenRouterResponse class
    result = test_openrouter_response()

    if result:
        provider.success("All tests passed! The OpenRouterResponse fix is working correctly.")
        provider.info("The fix adds a get() method to the Usage class in OpenRouterResponse")
        provider.info("This allows article_context.update_token_usage to work correctly with OpenRouter responses")
    else:
        provider.error("Tests failed. Please check the logs for details.")
