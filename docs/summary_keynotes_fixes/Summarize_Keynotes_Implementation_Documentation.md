# Summarize and Keynotes Enhancement Implementation

This document provides detailed information about the implementation of the enhancements to the Summarize and Keynotes sections of the article generation system.

## Questions and Answers

### Q1: Is the model that is specifically going to be used for summary and key takeaways and that is separate from the content generation model, is this model configurable through the config.py Config class and subsequently then through the main.py of both scripts?

**A1:** Yes, the separate model for summary and keynotes is configurable through the Config class in both scripts. I've added the following configuration parameters:

- `enable_separate_summary_model`: Boolean flag to enable/disable using a separate model
- `summary_keynotes_model`: String specifying the model to use (default is "anthropic/claude-3-opus-20240229")

These parameters can be set when initializing the Config class in main.py for both scripts. For example:

```python
config = Config(
    # Other parameters...
    enable_separate_summary_model=True,
    summary_keynotes_model="anthropic/claude-3-opus-20240229",
)
```

### Q2: For chunking, are we using tiktoken for counting the tokens or an average estimate?

**A2:** For chunking, we're using character-based chunking rather than token-based chunking. This is a simpler approach that doesn't require tiktoken for every operation. The chunking is based on the number of characters in the text, with a default chunk size of 8000 characters.

This approach was chosen for a few reasons:
1. It's more universal and doesn't depend on specific tokenization algorithms
2. It's faster since we don't need to tokenize the entire text
3. It's still effective for breaking down large articles into manageable chunks

However, when actually sending the content to the LLM APIs, we do use proper token counting:
- In script1, we use the encoding from the ArticleContext object (which uses tiktoken)
- In script2, we use the token counting methods provided by the ArticleContext

### Q3: Also, for models for which the window size isn't known, as may happen with OpenRouter where we might have more than 300 models, so then what happens?

**A3:** For models with unknown context window sizes (like many models available through OpenRouter), the implementation takes a conservative approach:

1. We use configurable chunk sizes (`summary_chunk_size` and `keynotes_chunk_size`) that can be adjusted based on the model being used.

2. The default values (8000 characters) are set conservatively to work with most models.

3. If a model has a smaller context window than expected, the chunking mechanism will still work - it will just create more chunks than might be necessary.

4. For models with very large context windows, you can increase these values in the configuration to take advantage of the larger context.

5. The system doesn't try to automatically detect the context window size of each model, as this information isn't consistently available across all APIs and models.

This approach ensures that the system will work with any model, regardless of its context window size, while still allowing you to optimize for specific models by adjusting the configuration parameters.

## Overview

The implementation adds the ability to use a separate model with a larger context window for generating article summaries and keynotes (block notes). It also implements chunking functionality to handle articles that are too large to process in a single pass.

## Implementation Details

### 1. Configuration Changes

Added new configuration parameters to both `script1/utils/config.py` and `script2/config.py`:

```python
# Summary and Keynotes Model Settings
enable_separate_summary_model: bool = False  # Enable using a separate model for summary and keynotes
summary_keynotes_model: str = "anthropic/claude-3-opus-20240229"  # Default model with large context window
summary_max_tokens: int = 800  # Maximum tokens for summary generation
keynotes_max_tokens: int = 300  # Maximum tokens for keynotes generation
summary_chunk_size: int = 8000  # Chunk size for summary generation when chunking is needed
keynotes_chunk_size: int = 8000  # Chunk size for keynotes generation when chunking is needed
```

These parameters can be set when initializing the Config class in main.py:

```python
config = Config(
    # Other parameters...
    enable_separate_summary_model=True,
    summary_keynotes_model="anthropic/claude-3-opus-20240229",
    summary_max_tokens=800,
    keynotes_max_tokens=300,
    summary_chunk_size=8000,
    keynotes_chunk_size=8000,
)
```

### 2. Chunking Utilities

Created chunking utility files for both scripts:
- `script1/article_generator/chunking_utils.py`
- `script2/article_generator/chunking_utils.py`

These files contain two main functions:

1. `chunk_article_for_processing`: Splits an article into chunks based on a specified chunk size
2. `combine_chunk_results`: Combines the results from processing multiple chunks

The chunking algorithm:
1. Checks if the article is small enough to process in one chunk
2. If not, splits the article into chunks, ensuring that:
   - Each chunk contains the article title for context
   - Sections are kept intact when possible
   - Large sections are split if necessary
   - There is some overlap between chunks for context continuity

### 3. Summary Generation Enhancement

Updated the summary generation functions to:
1. Check if a separate model should be used
2. Split the article into chunks if needed
3. Process each chunk with the appropriate model
4. Combine the results from all chunks

For script1, the changes were made in `script1/article_generator/content_generator.py`:
```python
def generate_article_summary(context, keyword, article_dict, summarize_prompt):
    # Check if separate model should be used
    use_separate_model = (
        hasattr(context.config, 'enable_separate_summary_model') and
        context.config.enable_separate_summary_model and
        hasattr(context.config, 'summary_keynotes_model') and
        context.config.summary_keynotes_model
    )

    # Chunk the article if needed
    article_chunks = chunk_article_for_processing(article_dict, chunk_size=chunk_size)

    # Process each chunk
    for chunk in article_chunks:
        # Generate summary with appropriate model
        if use_separate_model and context.config.use_openrouter:
            # Use OpenRouter with specified model
            chunk_summary = make_openrouter_api_call(...)
        else:
            # Use standard model
            chunk_summary = gpt_completion(...)

    # Combine results
    summary = combine_chunk_results(chunk_results)
    return summary
```

Similar changes were made for script2 in `script2/article_generator/generator.py`.

### 4. Keynotes Generation Enhancement

Updated the block notes (keynotes) generation functions with the same approach as the summary generation:
1. Check if a separate model should be used
2. Split the article into chunks if needed
3. Process each chunk with the appropriate model
4. Combine the results from all chunks

For script1, the changes were made in `script1/article_generator/text_processor.py`:
```python
def generate_block_notes(context, article_content, blocknote_prompt, *, engine, ...):
    # Check if separate model should be used
    use_separate_model = (
        hasattr(context.config, 'enable_separate_summary_model') and
        context.config.enable_separate_summary_model and
        hasattr(context.config, 'summary_keynotes_model') and
        context.config.summary_keynotes_model
    )

    # Chunk the article if needed
    article_chunks = chunk_article_for_processing(article_content, chunk_size=chunk_size)

    # Process each chunk
    for chunk in article_chunks:
        # Generate keynotes with appropriate model
        if use_separate_model and context.config.use_openrouter:
            # Use OpenRouter with specified model
            chunk_keynotes = make_openrouter_api_call(...)
        else:
            # Use standard model
            chunk_keynotes = openai.chat.completions.create(...)

    # Combine results
    block_notes = combine_chunk_results(chunk_results)
    return block_notes
```

Similar changes were made for script2 in `script2/article_generator/text_processor.py`.

## Running Tests

To test the chunking functionality, you can run the test script:

```bash
python tests/test_chunking.py
```

This script tests:
1. Chunking articles of different sizes
2. Combining results from multiple chunks

To test the full implementation with actual article generation:

1. Configure the system to use a separate model:
   ```python
   # In main.py
   config = Config(
       enable_separate_summary_model=True,
       summary_keynotes_model="anthropic/claude-3-opus-20240229",
       # Other parameters...
   )
   ```

2. Generate an article with a large content to test chunking:
   ```bash
   python script1/main.py --keyword "your test keyword"
   # or
   python script2/main.py --keyword "your test keyword"
   ```

3. Check the logs to verify that:
   - The correct model is being used for summary and keynotes generation
   - Chunking is applied when needed
   - The results are properly combined

## Handling Different Context Window Sizes

For models with unknown context window sizes, the system uses the configured chunk size as a safe default. You can adjust the `summary_chunk_size` and `keynotes_chunk_size` parameters based on your needs.

If you're using a model with a very large context window, you can set these values higher. If you're using a model with a smaller context window, you should set these values lower.

The chunking mechanism ensures that even if the model's context window is smaller than expected, the article will still be processed correctly by breaking it into smaller pieces.
