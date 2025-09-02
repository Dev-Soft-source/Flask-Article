# Error Handling Analysis and Recommendations

## Introduction

This document provides a comprehensive analysis of the error handling mechanisms in the article generation system, covering both script1 and script2. The goal is to identify strengths, weaknesses, and opportunities for improvement in how errors are handled, logged, and communicated to users.

Robust error handling is critical for an article generation system that relies on multiple external APIs, file operations, and complex processing steps. Effective error handling ensures:

1. **Resilience**: The system can recover from failures and continue operation
2. **Transparency**: Users understand what went wrong and how to fix it
3. **Maintainability**: Developers can quickly identify and fix issues
4. **User Experience**: Users are not frustrated by cryptic error messages or silent failures

## Current Error Handling Implementation

### Logger Implementation

Both script1 and script2 implement custom logging systems with similar capabilities:

**Script1**: Uses `RichLogger` class in `article_generator/logger.py`
- Implements `LoggerInterface` with methods for different log levels
- Uses Rich library for colorful console output
- Logs to both console and session log files
- Tracks statistics for various operations
- Provides specialized logging for URL access, content scraping, etc.

**Script2**: Uses `RAGLogger` class in `article_generator/logger.py`
- Extends `RichProvider` for colorful console output
- Similar functionality to Script1's logger
- Logs to both console and session log files
- Tracks statistics for various operations

### Error Handling in Main Scripts

**Script1**:
- Uses try-except blocks to catch and handle exceptions
- Provides user-friendly error messages for CSV parsing issues
- Continues processing other keywords when one fails
- Uses Rich console for colorful error output

**Script2**:
- More comprehensive error handling with detailed traceback information
- Uses Rich traceback handler with `show_locals=True`
- Structured error handling in the `process_articles` function
- Provides summary of successful and failed generations

### API Validation and Error Handling

Both scripts implement API validation for various services:

**Script1**:
- Validates OpenAI, YouTube, SerpAPI, and WordPress APIs
- Provides detailed error messages for API validation failures
- Continues with limited functionality when some APIs fail

**Script2**:
- Similar API validation with more structured approach
- Uses `APIValidator` class in `utils/api_utils.py`
- Provides specific error messages for different failure scenarios
- Continues with limited functionality when some APIs fail

### CSV Parsing and Validation

Both scripts use the `UnifiedCSVProcessor` for CSV parsing and validation:

- Detailed validation of CSV structure
- Colorful error messages for validation failures
- Flexible handling of different CSV formats
- Helpful error messages with examples of correct format

### Retry Mechanisms and Backoff Strategies

Both scripts implement retry mechanisms for API calls:

**Script1**:
- Uses `@retry` decorator from tenacity library
- Implements exponential backoff for API calls
- Configurable retry parameters

**Script2**:
- Similar retry mechanisms with tenacity
- More specific retry conditions using `retry_if_exception_type`
- Custom `RetryHandler` class in `utils/api_utils.py`

### Rate Limiting

Both scripts implement rate limiting to prevent API rate limit errors:

**Script1**:
- Uses `RateLimiter` class in `utils/rate_limiter.py`
- Implements token bucket algorithm
- Supports per-minute and per-day rate limits
- Configurable cooldown periods

**Script2**:
- Similar rate limiting implementation
- Integrated with API calls

## Strengths of Current Implementation

1. **Rich Logging**: Both scripts use the Rich library for colorful, structured console output that makes it easy to distinguish between different types of messages.

2. **Structured Error Messages**: Error messages are generally informative and provide context about what went wrong.

3. **Retry Mechanisms**: The use of retry decorators with exponential backoff helps handle transient API failures gracefully.

4. **Rate Limiting**: The implementation of rate limiting helps prevent API rate limit errors, which is crucial for systems that make many API calls.

5. **Fallback Mechanisms**: Some components have fallback mechanisms for critical failures, such as the content generator providing fallback text when API calls fail.

6. **Detailed Logging**: The system logs detailed information about operations, which helps with debugging and monitoring.

7. **Unified CSV Processing**: The unified CSV processor provides robust validation and helpful error messages for CSV parsing issues.

## Areas for Improvement

1. **Inconsistencies Between Scripts**: While both scripts have similar error handling approaches, there are inconsistencies in implementation details that could be standardized.

2. **Error Recovery Strategies**: Some components lack clear error recovery strategies, leading to potential cascading failures.

3. **User Feedback for Errors**: Error messages could be more actionable, providing clearer guidance on how to resolve issues.

4. **Error Categorization**: Errors are not consistently categorized by severity or type, making it difficult to prioritize and address them.

5. **Error Reporting**: There is limited centralized error reporting, making it challenging to monitor system health.

6. **Testing Error Handling**: There appears to be limited testing of error handling code, which could lead to unhandled edge cases.

7. **Graceful Degradation**: Some features fail completely when dependencies are unavailable, rather than degrading gracefully.

8. **Consistent Traceback Handling**: Traceback information is handled differently across components, with some providing detailed tracebacks and others not.

9. **Environment-Specific Error Handling**: There is limited adaptation of error handling based on the environment (development vs. production).

10. **Error Documentation**: Error codes and common issues are not well-documented for users.

## Recommendations

### 1. Standardize Error Handling Across Both Scripts

**Short-term Improvements**:

- Create a shared error handling module that both scripts can use
- Standardize error message formats across all components
- Ensure consistent use of traceback information

**Implementation Example**:

```python
# shared/error_handler.py
from typing import Optional, Dict, Any
import traceback
from rich.console import Console
from rich.panel import Panel

console = Console()

def handle_error(error: Exception, context: Dict[str, Any] = None,
                 show_traceback: bool = True, fatal: bool = False) -> None:
    """
    Standardized error handler for both scripts.

    Args:
        error: The exception that occurred
        context: Additional context about where the error occurred
        show_traceback: Whether to show the traceback
        fatal: Whether this is a fatal error that should stop execution
    """
    # Format error message
    error_type = type(error).__name__
    error_message = str(error)

    # Add context information
    context_str = ""
    if context:
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])

    # Create panel with error information
    panel_content = f"[bold red]{error_type}[/]: {error_message}"
    if context_str:
        panel_content += f"\n\n[yellow]Context:[/]\n{context_str}"

    if show_traceback:
        trace = traceback.format_exc()
        if trace and trace != "NoneType: None\n":
            panel_content += f"\n\n[dim red]Traceback:[/]\n{trace}"

    console.print(Panel(
        panel_content,
        title="[bold red]ERROR[/]",
        border_style="red"
    ))

    # If fatal, exit
    if fatal:
        console.print("[bold red]Fatal error. Exiting...[/]")
        exit(1)
```

### 2. Implement More Robust Error Recovery Strategies

**Medium-term Improvements**:

- Implement circuit breaker pattern for external API calls
- Add more fallback mechanisms for critical components
- Implement automatic retry with different models/APIs when primary ones fail

**Implementation Example**:

```python
# Circuit breaker implementation
class CircuitBreaker:
    def __init__(self, name, failure_threshold=5, reset_timeout=60):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
        self.last_failure_time = 0

    def execute(self, func, *args, **kwargs):
        """Execute function with circuit breaker pattern."""
        current_time = time.time()

        # Check if circuit is OPEN
        if self.state == "OPEN":
            # Check if reset timeout has elapsed
            if current_time - self.last_failure_time > self.reset_timeout:
                self.state = "HALF-OPEN"
                logger.info(f"Circuit {self.name} changed from OPEN to HALF-OPEN")
            else:
                logger.warning(f"Circuit {self.name} is OPEN. Skipping call.")
                raise CircuitBreakerOpenError(f"Circuit {self.name} is OPEN")

        try:
            result = func(*args, **kwargs)

            # If successful and in HALF-OPEN, close the circuit
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failures = 0
                logger.info(f"Circuit {self.name} changed from HALF-OPEN to CLOSED")

            return result

        except Exception as e:
            # Increment failure count
            self.failures += 1
            self.last_failure_time = current_time

            # Check if failure threshold reached
            if self.failures >= self.failure_threshold and self.state == "CLOSED":
                self.state = "OPEN"
                logger.warning(f"Circuit {self.name} changed from CLOSED to OPEN after {self.failures} failures")

            # Re-raise the exception
            raise e
```

### 3. Improve User Feedback for Errors

**Short-term Improvements**:

- Categorize errors by user actionability
- Provide clear, actionable guidance for user-fixable errors
- Use consistent formatting for error messages

**Implementation Example**:

```python
def validate_csv_file(file_path):
    """Validate CSV file with improved user feedback."""
    try:
        # Existing validation code...
        pass
    except FileNotFoundError:
        console.print(Panel(
            "[bold red]File not found.[/]\n\n"
            "[yellow]Action required:[/]\n"
            "1. Check that the file exists at the specified path\n"
            "2. Ensure you have permission to read the file\n"
            f"3. The current working directory is: {os.getcwd()}\n"
            f"4. The file path you provided was: {file_path}",
            title="[bold red]FILE ERROR[/]",
            border_style="red"
        ))
        return False
    except csv.Error as e:
        console.print(Panel(
            f"[bold red]CSV parsing error: {str(e)}[/]\n\n"
            "[yellow]Action required:[/]\n"
            "1. Check that the file is a valid CSV file\n"
            "2. Ensure the file is not corrupted\n"
            "3. Try opening the file in a spreadsheet application to verify its contents\n"
            "4. Save the file with UTF-8 encoding",
            title="[bold red]CSV ERROR[/]",
            border_style="red"
        ))
        return False
    # More specific error handlers...
```

### 4. Categorize and Prioritize Errors

**Medium-term Improvements**:

- Define error severity levels (critical, error, warning, info)
- Implement error codes for common errors
- Create an error catalog with descriptions and solutions

**Implementation Example**:

```python
# Error catalog
ERROR_CATALOG = {
    "API001": {
        "message": "OpenAI API key is invalid or expired",
        "severity": "CRITICAL",
        "description": "The system cannot authenticate with the OpenAI API",
        "solution": "Check your OpenAI API key in the .env file and ensure it is valid and has sufficient credits"
    },
    "API002": {
        "message": "OpenAI API rate limit exceeded",
        "severity": "ERROR",
        "description": "The system has made too many requests to the OpenAI API in a short period",
        "solution": "Wait a few minutes and try again, or reduce the number of concurrent requests"
    },
    # More error definitions...
}

def log_error_with_code(code, context=None):
    """Log an error using the error catalog."""
    if code not in ERROR_CATALOG:
        logger.error(f"Unknown error code: {code}")
        return

    error_info = ERROR_CATALOG[code]
    severity = error_info["severity"]
    message = error_info["message"]
    description = error_info["description"]
    solution = error_info["solution"]

    if severity == "CRITICAL":
        logger.error(f"[{code}] {message}")
    elif severity == "ERROR":
        logger.error(f"[{code}] {message}")
    elif severity == "WARNING":
        logger.warning(f"[{code}] {message}")

    if context:
        logger.debug(f"Error context: {context}")

    logger.info(f"Description: {description}")
    logger.info(f"Solution: {solution}")
```

### 5. Implement Error Reporting and Monitoring

**Long-term Improvements**:

- Add centralized error logging to a file or database
- Implement error aggregation and analysis
- Create a dashboard for monitoring error rates and patterns

**Implementation Example**:

```python
# Error reporting module
class ErrorReporter:
    def __init__(self, log_dir="logs/errors"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.error_counts = {}

    def report_error(self, error_type, error_message, context=None):
        """Report an error to the centralized system."""
        # Increment error count
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        # Log to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.log_dir, f"error_{timestamp}.json")

        error_data = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_message,
            "context": context or {}
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(error_data, f, indent=2)

        # Check if this is a high-frequency error
        if self.error_counts[error_type] > 10:
            logger.warning(f"High frequency of {error_type} errors: {self.error_counts[error_type]} occurrences")

    def get_error_summary(self):
        """Get a summary of errors reported."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_types": self.error_counts
        }
```

### 6. Test Error Handling with Unit Tests

**Medium-term Improvements**:

- Create unit tests specifically for error handling code
- Simulate various error conditions to test recovery
- Verify that error messages are helpful and accurate

**Implementation Example**:

```python
# Example test for API error handling
def test_openai_api_error_handling():
    """Test handling of OpenAI API errors."""
    # Mock the OpenAI API to simulate an error
    with patch("openai.ChatCompletion.create") as mock_create:
        # Simulate a rate limit error
        mock_create.side_effect = openai.error.RateLimitError(
            "Rate limit exceeded",
            response=MagicMock(status_code=429)
        )

        # Call the function that uses the API
        result = generate_completion("Test prompt", "gpt-3.5-turbo")

        # Verify that it handled the error gracefully
        assert "Error generating response" in result
        assert mock_create.call_count == 3  # Should retry 3 times
```

## Implementation Plan

### Short-term Improvements (1-2 weeks)

1. **Standardize Error Messages**:
   - Create shared error message formats
   - Ensure consistent use of Rich for formatting
   - Standardize traceback handling

2. **Improve CSV Validation Feedback**:
   - Enhance error messages with more specific guidance
   - Add examples of correct format in error messages
   - Implement validation previews that show problematic rows

3. **Enhance API Error Handling**:
   - Standardize API error handling across both scripts
   - Improve error messages for API failures
   - Add more context to API error messages

### Medium-term Improvements (2-4 weeks)

1. **Implement Error Categorization**:
   - Define error severity levels
   - Create error codes for common errors
   - Implement error catalog

2. **Add Error Recovery Strategies**:
   - Implement circuit breaker pattern
   - Add more fallback mechanisms
   - Implement automatic retry with different models/APIs

3. **Create Error Handling Tests**:
   - Write unit tests for error handling code
   - Simulate various error conditions
   - Verify error recovery behavior

### Long-term Improvements (1-2 months)

1. **Implement Error Reporting System**:
   - Create centralized error logging
   - Implement error aggregation and analysis
   - Build monitoring dashboard

2. **Enhance Graceful Degradation**:
   - Redesign components to work with limited functionality
   - Implement feature flags for problematic components
   - Add configuration options for fallback behavior

3. **Create Comprehensive Error Documentation**:
   - Document all error codes and messages
   - Create troubleshooting guide
   - Add examples of common errors and solutions

## Specific Improvement Areas

### CSV Parsing Error Handling

The CSV parsing is a critical component that can benefit from enhanced error handling:

#### Current Implementation

- The `UnifiedCSVProcessor` provides validation and error messages for CSV files
- It detects headers, validates required columns, and checks data rows
- Error messages are displayed using Rich library with colorful formatting
- The processor can handle both simple keyword lists and structured CSV files

#### Recommended Improvements

1. **More Specific Error Messages**:
   - Provide line numbers for problematic rows
   - Show examples of the actual problematic data
   - Offer suggestions for fixing common issues

2. **Progressive Validation**:
   - Validate the file in stages and report all issues at once
   - Check file existence, format, headers, and data separately
   - Provide a summary of all validation issues

3. **Interactive Fixing**:
   - Offer to fix simple issues automatically
   - Provide a preview of how the fix would look
   - Allow users to confirm or reject the fix

**Implementation Example**:

```python
def validate_csv_file(file_path):
    """Enhanced CSV validation with progressive reporting."""
    issues = []

    # Stage 1: Check file existence
    if not os.path.exists(file_path):
        issues.append({
            "stage": "file_existence",
            "message": f"File not found: {file_path}",
            "suggestion": f"Check that the file exists at {os.path.abspath(file_path)}"
        })
        return False, issues

    # Stage 2: Check file format
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(1024)
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
    except Exception as e:
        issues.append({
            "stage": "file_format",
            "message": f"Not a valid CSV file: {str(e)}",
            "suggestion": "Ensure the file is a properly formatted CSV file"
        })
        return False, issues

    # Stage 3: Check headers
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, dialect)
            headers = next(reader) if has_header else []

            if has_header:
                required_columns = ["keyword"]
                missing_columns = [col for col in required_columns if col.lower() not in [h.lower() for h in headers]]

                if missing_columns:
                    issues.append({
                        "stage": "headers",
                        "message": f"Missing required columns: {', '.join(missing_columns)}",
                        "suggestion": f"Ensure your CSV has these columns: {', '.join(required_columns)}"
                    })
    except Exception as e:
        issues.append({
            "stage": "headers",
            "message": f"Error checking headers: {str(e)}",
            "suggestion": "Check that the file is not corrupted"
        })
        return False, issues

    # Stage 4: Check data rows
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, dialect)
            if has_header:
                next(reader)  # Skip header

            for i, row in enumerate(reader, start=1):
                if not row or (len(row) == 1 and not row[0].strip()):
                    issues.append({
                        "stage": "data",
                        "message": f"Empty row at line {i+1}",
                        "suggestion": "Remove empty rows or fill with data"
                    })
                elif has_header and len(row) != len(headers):
                    issues.append({
                        "stage": "data",
                        "message": f"Row {i+1} has {len(row)} columns but header has {len(headers)} columns",
                        "suggestion": "Ensure all rows have the same number of columns as the header"
                    })
    except Exception as e:
        issues.append({
            "stage": "data",
            "message": f"Error checking data rows: {str(e)}",
            "suggestion": "Check that the file is not corrupted"
        })
        return False, issues

    # Return validation result
    is_valid = len(issues) == 0
    return is_valid, issues
```

### API Error Handling

API calls are another critical area that can benefit from improved error handling:

#### Current Implementation

- Both scripts validate API keys before using them
- They use retry mechanisms with exponential backoff
- Rate limiting is implemented to prevent API rate limit errors
- Some components have fallback mechanisms for API failures

#### Recommended Improvements

1. **Unified API Client**:
   - Create a unified API client for each service
   - Standardize error handling across all API calls
   - Implement consistent retry and backoff strategies

2. **Smart Fallbacks**:
   - Implement fallbacks to alternative models or services
   - Degrade functionality gracefully when APIs are unavailable
   - Cache previous successful responses for critical features

3. **Proactive Monitoring**:
   - Monitor API health and quota usage
   - Warn users before they hit rate limits
   - Provide estimates of remaining quota

### LLM-Specific Error Handling

The article generation system relies heavily on LLM APIs (OpenAI, OpenRouter), which have unique error patterns:

#### Current Implementation

- Both scripts use retry mechanisms for LLM API calls
- They handle rate limiting errors with exponential backoff
- Some components have fallback text for critical failures
- Token tracking is implemented to avoid context window limits

#### Recommended Improvements

1. **Content Validation**:
   - Validate LLM outputs against expected formats
   - Detect and handle hallucinations or off-topic responses
   - Implement fallback strategies for low-quality outputs

2. **Token Management**:
   - Implement more sophisticated token counting and chunking
   - Dynamically adjust prompts based on available token space
   - Provide clear error messages for token limit issues

3. **Model-Specific Error Handling**:
   - Customize error handling for different LLM models
   - Handle model-specific quirks and limitations
   - Implement model-specific fallback strategies

**Implementation Example**:

```python
class LLMOutputValidator:
    """Validates and handles issues with LLM outputs."""

    def __init__(self, expected_format=None, min_length=50, max_length=10000):
        self.expected_format = expected_format
        self.min_length = min_length
        self.max_length = max_length
        self.fallback_responses = {}

    def add_fallback(self, content_type, fallback_text):
        """Add a fallback response for a specific content type."""
        self.fallback_responses[content_type] = fallback_text

    def validate(self, content, content_type=None):
        """
        Validate LLM output and return validation result.

        Args:
            content (str): The content to validate
            content_type (str): Type of content (e.g., 'introduction', 'paragraph', etc.)

        Returns:
            dict: Validation result with status and issues
        """
        issues = []

        # Check for empty or very short content
        if not content or len(content) < self.min_length:
            issues.append({
                "type": "length",
                "severity": "error",
                "message": f"Content is too short ({len(content) if content else 0} chars)"
            })

        # Check for excessively long content
        if content and len(content) > self.max_length:
            issues.append({
                "type": "length",
                "severity": "warning",
                "message": f"Content is very long ({len(content)} chars)"
            })

        # Check for expected format if specified
        if self.expected_format and content:
            if self.expected_format == "html" and not self._is_valid_html(content):
                issues.append({
                    "type": "format",
                    "severity": "error",
                    "message": "Content is not valid HTML"
                })
            elif self.expected_format == "markdown" and not self._is_valid_markdown(content):
                issues.append({
                    "type": "format",
                    "severity": "warning",
                    "message": "Content may not be valid Markdown"
                })

        # Check for hallucination indicators
        if content:
            hallucination_score = self._check_hallucination(content)
            if hallucination_score > 0.7:
                issues.append({
                    "type": "hallucination",
                    "severity": "error",
                    "message": f"Content likely contains hallucinations (score: {hallucination_score:.2f})"
                })
            elif hallucination_score > 0.4:
                issues.append({
                    "type": "hallucination",
                    "severity": "warning",
                    "message": f"Content may contain hallucinations (score: {hallucination_score:.2f})"
                })

        # Determine overall status
        has_error = any(issue["severity"] == "error" for issue in issues)
        has_warning = any(issue["severity"] == "warning" for issue in issues)

        status = "valid"
        if has_error:
            status = "invalid"
        elif has_warning:
            status = "warning"

        return {
            "status": status,
            "issues": issues,
            "content": content
        }

    def handle_invalid_output(self, validation_result, content_type=None, retry_func=None):
        """
        Handle invalid LLM output based on validation result.

        Args:
            validation_result (dict): Validation result from validate()
            content_type (str): Type of content
            retry_func (callable): Function to retry generation

        Returns:
            str: Fixed or fallback content
        """
        if validation_result["status"] == "valid":
            return validation_result["content"]

        # Log issues
        for issue in validation_result["issues"]:
            if issue["severity"] == "error":
                logger.error(f"LLM output issue: {issue['message']}")
            else:
                logger.warning(f"LLM output issue: {issue['message']}")

        # For warnings, we can still use the content but log the issues
        if validation_result["status"] == "warning":
            logger.warning(f"Using content with warnings for {content_type}")
            return validation_result["content"]

        # For errors, try to retry or use fallback
        if retry_func and self._should_retry(validation_result["issues"]):
            logger.info(f"Retrying generation for {content_type}")
            new_content = retry_func()
            new_validation = self.validate(new_content, content_type)

            # If retry succeeded, use the new content
            if new_validation["status"] != "invalid":
                return new_content

        # Use fallback if available
        if content_type in self.fallback_responses:
            logger.warning(f"Using fallback content for {content_type}")
            return self.fallback_responses[content_type]

        # Last resort: try to fix the content
        logger.warning(f"Attempting to fix invalid content for {content_type}")
        return self._fix_content(validation_result["content"], validation_result["issues"])

    def _is_valid_html(self, content):
        """Check if content is valid HTML."""
        # Simple check for balanced tags
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
            return True
        except Exception:
            return False

    def _is_valid_markdown(self, content):
        """Check if content is valid Markdown."""
        # Simple check for markdown structure
        import re
        # Check for common markdown elements
        has_headers = bool(re.search(r'^#{1,6}\s', content, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[\*\-\+]\s', content, re.MULTILINE))
        has_paragraphs = len(content.split('\n\n')) > 1

        return has_headers or has_lists or has_paragraphs

    def _check_hallucination(self, content):
        """
        Check for indicators of hallucination in content.
        Returns a score between 0 and 1, where higher means more likely hallucination.
        """
        # This is a simplified example - in practice, you might use more sophisticated methods
        import re

        # Check for phrases that often indicate hallucination
        hallucination_phrases = [
            r"I don't have (specific|current|up-to-date) information",
            r"As of my last (update|training)",
            r"I cannot browse the (internet|web)",
            r"I don't have access to (real-time|current) data",
            r"I cannot verify",
            r"I'm not able to search",
        ]

        # Count matches
        match_count = 0
        for phrase in hallucination_phrases:
            if re.search(phrase, content, re.IGNORECASE):
                match_count += 1

        # Calculate score (simplified)
        score = min(1.0, match_count / 3)  # 3 or more matches -> score of 1.0

        return score

    def _should_retry(self, issues):
        """Determine if we should retry based on the issues."""
        # Retry for length issues or hallucinations, but not format issues
        # (which are less likely to be fixed by simply retrying)
        return any(issue["type"] in ["length", "hallucination"] for issue in issues)

    def _fix_content(self, content, issues):
        """Attempt to fix content based on issues."""
        fixed_content = content

        for issue in issues:
            if issue["type"] == "format" and issue["message"] == "Content is not valid HTML":
                # Try to fix HTML
                try:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(fixed_content, "html.parser")
                    fixed_content = str(soup)
                except Exception:
                    pass

        return fixed_content

# Example usage in content generation
def generate_article_section(keyword, section_type, prompt, model="gpt-4"):
    """Generate an article section with enhanced error handling."""
    # Initialize validator with appropriate expectations for this section type
    if section_type == "introduction":
        validator = LLMOutputValidator(
            expected_format="markdown",
            min_length=200,
            max_length=1000
        )
        validator.add_fallback(
            "introduction",
            f"Welcome to our guide on {keyword}. In this article, we'll explore "
            f"everything you need to know about this topic."
        )
    elif section_type == "conclusion":
        validator = LLMOutputValidator(
            expected_format="markdown",
            min_length=150,
            max_length=800
        )
        validator.add_fallback(
            "conclusion",
            f"In conclusion, {keyword} is an important topic worth understanding. "
            f"We hope this guide has been helpful in your journey."
        )
    else:
        validator = LLMOutputValidator(expected_format="markdown")
        validator.add_fallback(
            section_type,
            f"This section provides information about {keyword} related to {section_type}."
        )

    # Define retry function
    def retry_generation():
        try:
            # Try with a different model or more explicit prompt
            fallback_model = "gpt-3.5-turbo" if model == "gpt-4" else "gpt-4"
            enhanced_prompt = f"{prompt}\n\nIMPORTANT: Please provide a detailed, accurate, and well-structured response."
            return make_llm_call(enhanced_prompt, fallback_model)
        except Exception as e:
            logger.error(f"Retry generation failed: {str(e)}")
            return None

    try:
        # Make the initial LLM call
        content = make_llm_call(prompt, model)

        # Validate the output
        validation_result = validator.validate(content, section_type)

        # Handle any issues
        return validator.handle_invalid_output(
            validation_result,
            content_type=section_type,
            retry_func=retry_generation
        )
    except Exception as e:
        logger.error(f"Error generating {section_type}: {str(e)}")
        # Use fallback content in case of exception
        return validator.fallback_responses.get(
            section_type,
            f"Information about {keyword} related to {section_type}."
        )
```

## Conclusion

The article generation system has a solid foundation for error handling, with rich logging, retry mechanisms, and rate limiting. However, there are opportunities to improve consistency, error recovery, user feedback, and monitoring.

By implementing the recommendations in this document, the system can become more resilient, user-friendly, and maintainable. The short-term improvements will address the most pressing issues, while the medium and long-term improvements will create a more comprehensive error handling system.

The next steps should be to prioritize these recommendations based on the most common and impactful errors encountered in production, and to create a detailed implementation plan for each improvement.
