# Implementation Plan for Summarize and Keynotes Enhancement

## Overview

This implementation plan outlines the changes needed to enhance the Summarize and Keynotes sections of the article generation system. The key requirements are:

1. Use a large context window LLM model for processing the entire article
2. Maintain different prompts for Summarize and Keynotes sections
3. Implement chunking for very large articles that exceed the context window

The implementation is divided into the following parts:
1. Configuration Changes
2. Utility Functions for Chunking
3. Summary Generation Enhancement
4. Keynotes Generation Enhancement

## Part 1: Configuration Changes

### Script 1 Configuration Changes

#### File: `script1/utils/config.py`

Add the following parameters to the `Config` class:

```python
# Summary and Keynotes Model Settings
enable_separate_summary_model: bool = False  # Enable using a separate model for summary and keynotes
summary_keynotes_model: str = "anthropic/claude-3-opus-20240229"  # Default model with large context window
summary_max_tokens: int = 800  # Maximum tokens for summary generation
keynotes_max_tokens: int = 300  # Maximum tokens for keynotes generation
summary_chunk_size: int = 8000  # Chunk size for summary generation when chunking is needed
keynotes_chunk_size: int = 8000  # Chunk size for keynotes generation when chunking is needed
```

#### File: `script1/main.py`

Update the Config initialization to include the new parameters:

```python
# Add these parameters to the Config initialization
enable_separate_summary_model=True,
summary_keynotes_model="anthropic/claude-3-opus-20240229",  # Or another large context window model
summary_max_tokens=800,
keynotes_max_tokens=300,
summary_chunk_size=8000,
keynotes_chunk_size=8000,
```

### Script 2 Configuration Changes

#### File: `script2/config.py`

Add the following parameters to the `Config` class:

```python
# Summary and Keynotes Model Settings
enable_separate_summary_model: bool = False  # Enable using a separate model for summary and keynotes
summary_keynotes_model: str = "anthropic/claude-3-opus-20240229"  # Default model with large context window
summary_max_tokens: int = 800  # Maximum tokens for summary generation
keynotes_max_tokens: int = 300  # Maximum tokens for keynotes generation
summary_chunk_size: int = 8000  # Chunk size for summary generation when chunking is needed
keynotes_chunk_size: int = 8000  # Chunk size for keynotes generation when chunking is needed
```

#### File: `script2/main.py`

Update the Config initialization to include the new parameters:

```python
# Add these parameters to the Config initialization
enable_separate_summary_model=True,
summary_keynotes_model="anthropic/claude-3-opus-20240229",  # Or another large context window model
summary_max_tokens=800,
keynotes_max_tokens=300,
summary_chunk_size=8000,
keynotes_chunk_size=8000,
```

## Part 2: Utility Functions for Chunking

We need to create utility functions to handle chunking of large articles. These functions will be used by both the summary and keynotes generation functions.

### Script 1: Create a new file `script1/article_generator/chunking_utils.py`

```python
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from typing import Dict, List, Any
from article_generator.logger import logger

def chunk_article_for_processing(
    article_dict: Dict[str, Any],
    chunk_size: int = 8000,
    overlap: int = 500
) -> List[Dict[str, Any]]:
    """
    Split an article into chunks for processing by LLMs with limited context windows.

    Args:
        article_dict: Dictionary containing article parts (title, introduction, sections, conclusion)
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of article chunk dictionaries
    """
    logger.info(f"Chunking article for processing (chunk_size={chunk_size}, overlap={overlap})")

    # If the article is small enough, return it as is
    full_content = (
        f"Title: {article_dict.get('title', '')}\n\n"
        f"Introduction: {article_dict.get('introduction', '')}\n\n"
    )

    # Add sections
    if isinstance(article_dict.get('sections', []), list):
        for i, section in enumerate(article_dict.get('sections', [])):
            full_content += f"Section {i+1}: {section}\n\n"

    # Add conclusion
    full_content += f"Conclusion: {article_dict.get('conclusion', '')}"

    # If content is small enough, return as single chunk
    if len(full_content) <= chunk_size:
        logger.info("Article is small enough to process in one chunk")
        return [article_dict]

    # Otherwise, split into chunks
    logger.info(f"Article is too large ({len(full_content)} chars), splitting into chunks")

    chunks = []
    current_chunk = {
        "title": article_dict.get('title', ''),
        "introduction": "",
        "sections": [],
        "conclusion": ""
    }
    current_size = len(article_dict.get('title', ''))

    # Always include the title in each chunk

    # Add introduction if it fits
    intro = article_dict.get('introduction', '')
    if current_size + len(intro) <= chunk_size:
        current_chunk["introduction"] = intro
        current_size += len(intro)
    else:
        # Split introduction if needed
        current_chunk["introduction"] = intro[:chunk_size - current_size]
        logger.warning("Introduction had to be split across chunks")

    # Add sections
    if isinstance(article_dict.get('sections', []), list):
        for section in article_dict.get('sections', []):
            if current_size + len(section) <= chunk_size:
                current_chunk["sections"].append(section)
                current_size += len(section)
            else:
                # If this section would exceed chunk size, start a new chunk
                if current_chunk["introduction"] or current_chunk["sections"]:
                    chunks.append(current_chunk)

                    # Start new chunk with overlap
                    overlap_sections = []
                    if current_chunk["sections"]:
                        # Include the last section from previous chunk for context
                        overlap_sections = current_chunk["sections"][-1:]

                    current_chunk = {
                        "title": article_dict.get('title', ''),
                        "introduction": "",
                        "sections": overlap_sections,
                        "conclusion": ""
                    }
                    current_size = len(article_dict.get('title', '')) + sum(len(s) for s in overlap_sections)

                # Now add the current section
                if len(section) <= chunk_size:
                    current_chunk["sections"].append(section)
                    current_size += len(section)
                else:
                    # If the section itself is too large, split it
                    logger.warning(f"Section is too large ({len(section)} chars), splitting it")
                    section_parts = []
                    for i in range(0, len(section), chunk_size - overlap):
                        part = section[i:i + chunk_size - overlap]
                        section_parts.append(part)

                    # Add first part to current chunk
                    current_chunk["sections"].append(section_parts[0])
                    current_size += len(section_parts[0])

                    # Create new chunks for remaining parts
                    for part in section_parts[1:]:
                        chunks.append(current_chunk)
                        current_chunk = {
                            "title": article_dict.get('title', ''),
                            "introduction": "",
                            "sections": [part],
                            "conclusion": ""
                        }
                        current_size = len(article_dict.get('title', '')) + len(part)

    # Add conclusion to the last chunk if it fits
    conclusion = article_dict.get('conclusion', '')
    if current_size + len(conclusion) <= chunk_size:
        current_chunk["conclusion"] = conclusion
    else:
        # Add current chunk and create a final one for conclusion
        chunks.append(current_chunk)
        current_chunk = {
            "title": article_dict.get('title', ''),
            "introduction": "",
            "sections": [],
            "conclusion": conclusion
        }

    # Add the final chunk
    chunks.append(current_chunk)

    logger.success(f"Split article into {len(chunks)} chunks")
    return chunks

def combine_chunk_results(results: List[str]) -> str:
    """
    Combine results from processing multiple chunks.

    Args:
        results: List of text results from processing each chunk

    Returns:
        Combined result
    """
    logger.info(f"Combining results from {len(results)} chunks")

    if len(results) == 1:
        return results[0]

    # For multiple chunks, we need to combine them intelligently
    combined = ""

    for i, result in enumerate(results):
        if i == 0:
            # Use the first chunk as is
            combined = result
        else:
            # For subsequent chunks, try to avoid repetition
            # This is a simple approach - in practice, you might want more sophisticated merging
            sentences = result.split('. ')
            new_content = []

            for sentence in sentences:
                # Only add sentences that aren't substantially contained in the combined result
                if sentence and sentence.strip() and sentence.strip() + '.' not in combined:
                    new_content.append(sentence)

            if new_content:
                combined += " " + ". ".join(new_content)
                if not combined.endswith('.'):
                    combined += '.'

    logger.success(f"Combined chunk results (length: {len(combined)})")
    return combined
```

### Script 2: Create a new file `script2/article_generator/chunking_utils.py`

```python
# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from typing import Dict, List, Any
from article_generator.logger import logger

def chunk_article_for_processing(
    article_dict: Dict[str, Any],
    chunk_size: int = 8000,
    overlap: int = 500
) -> List[Dict[str, Any]]:
    """
    Split an article into chunks for processing by LLMs with limited context windows.

    Args:
        article_dict: Dictionary containing article parts (title, introduction, sections, conclusion)
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of article chunk dictionaries
    """
    logger.info(f"Chunking article for processing (chunk_size={chunk_size}, overlap={overlap})")

    # If the article is small enough, return it as is
    full_content = (
        f"Title: {article_dict.get('title', '')}\n\n"
        f"Introduction: {article_dict.get('introduction', '')}\n\n"
    )

    # Add sections
    if isinstance(article_dict.get('sections', []), list):
        for i, section in enumerate(article_dict.get('sections', [])):
            full_content += f"Section {i+1}: {section}\n\n"

    # Add conclusion
    full_content += f"Conclusion: {article_dict.get('conclusion', '')}"

    # If content is small enough, return as single chunk
    if len(full_content) <= chunk_size:
        logger.info("Article is small enough to process in one chunk")
        return [article_dict]

    # Otherwise, split into chunks
    logger.info(f"Article is too large ({len(full_content)} chars), splitting into chunks")

    chunks = []
    current_chunk = {
        "title": article_dict.get('title', ''),
        "introduction": "",
        "sections": [],
        "conclusion": ""
    }
    current_size = len(article_dict.get('title', ''))

    # Always include the title in each chunk

    # Add introduction if it fits
    intro = article_dict.get('introduction', '')
    if current_size + len(intro) <= chunk_size:
        current_chunk["introduction"] = intro
        current_size += len(intro)
    else:
        # Split introduction if needed
        current_chunk["introduction"] = intro[:chunk_size - current_size]
        logger.warning("Introduction had to be split across chunks")

    # Add sections
    if isinstance(article_dict.get('sections', []), list):
        for section in article_dict.get('sections', []):
            if current_size + len(section) <= chunk_size:
                current_chunk["sections"].append(section)
                current_size += len(section)
            else:
                # If this section would exceed chunk size, start a new chunk
                if current_chunk["introduction"] or current_chunk["sections"]:
                    chunks.append(current_chunk)

                    # Start new chunk with overlap
                    overlap_sections = []
                    if current_chunk["sections"]:
                        # Include the last section from previous chunk for context
                        overlap_sections = current_chunk["sections"][-1:]

                    current_chunk = {
                        "title": article_dict.get('title', ''),
                        "introduction": "",
                        "sections": overlap_sections,
                        "conclusion": ""
                    }
                    current_size = len(article_dict.get('title', '')) + sum(len(s) for s in overlap_sections)

                # Now add the current section
                if len(section) <= chunk_size:
                    current_chunk["sections"].append(section)
                    current_size += len(section)
                else:
                    # If the section itself is too large, split it
                    logger.warning(f"Section is too large ({len(section)} chars), splitting it")
                    section_parts = []
                    for i in range(0, len(section), chunk_size - overlap):
                        part = section[i:i + chunk_size - overlap]
                        section_parts.append(part)

                    # Add first part to current chunk
                    current_chunk["sections"].append(section_parts[0])
                    current_size += len(section_parts[0])

                    # Create new chunks for remaining parts
                    for part in section_parts[1:]:
                        chunks.append(current_chunk)
                        current_chunk = {
                            "title": article_dict.get('title', ''),
                            "introduction": "",
                            "sections": [part],
                            "conclusion": ""
                        }
                        current_size = len(article_dict.get('title', '')) + len(part)

    # Add conclusion to the last chunk if it fits
    conclusion = article_dict.get('conclusion', '')
    if current_size + len(conclusion) <= chunk_size:
        current_chunk["conclusion"] = conclusion
    else:
        # Add current chunk and create a final one for conclusion
        chunks.append(current_chunk)
        current_chunk = {
            "title": article_dict.get('title', ''),
            "introduction": "",
            "sections": [],
            "conclusion": conclusion
        }

    # Add the final chunk
    chunks.append(current_chunk)

    logger.success(f"Split article into {len(chunks)} chunks")
    return chunks

def combine_chunk_results(results: List[str]) -> str:
    """
    Combine results from processing multiple chunks.

    Args:
        results: List of text results from processing each chunk

    Returns:
        Combined result
    """
    logger.info(f"Combining results from {len(results)} chunks")

    if len(results) == 1:
        return results[0]

    # For multiple chunks, we need to combine them intelligently
    combined = ""

    for i, result in enumerate(results):
        if i == 0:
            # Use the first chunk as is
            combined = result
        else:
            # For subsequent chunks, try to avoid repetition
            # This is a simple approach - in practice, you might want more sophisticated merging
            sentences = result.split('. ')
            new_content = []

            for sentence in sentences:
                # Only add sentences that aren't substantially contained in the combined result
                if sentence and sentence.strip() and sentence.strip() + '.' not in combined:
                    new_content.append(sentence)

            if new_content:
                combined += " " + ". ".join(new_content)
                if not combined.endswith('.'):
                    combined += '.'

    logger.success(f"Combined chunk results (length: {len(combined)})")
    return combined
```

## Part 3: Summary Generation Enhancement

Now we need to modify the summary generation functions to use the specified model and implement chunking for large articles.

### Script 1: Modify `script1/article_generator/content_generator.py`

Find the `generate_article_summary` function and replace it with the following:

```python
def generate_article_summary(
    context: ArticleContext,
    keyword: str,
    article_dict: Dict[str, str],
    summarize_prompt: str,
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
        from article_generator.chunking_utils import chunk_article_for_processing, combine_chunk_results

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
                f"Main Content:\n{'\n'.join(chunk.get('sections', []))}\n\n"
                f"Conclusion: {chunk.get('conclusion', '')}"
            )

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
                response = make_openrouter_api_call(
                    messages=messages,
                    model=context.config.summary_keynotes_model,
                    api_key=context.config.openrouter_api_key,
                    site_url=context.config.openrouter_site_url,
                    site_name=context.config.openrouter_site_name,
                    temperature=0.7,
                    max_tokens=max_tokens,
                    seed=seed
                )

                chunk_summary = response.choices[0].message.content.strip()
            else:
                # Use the standard gpt_completion function
                chunk_summary = gpt_completion(
                    context,
                    prompt,
                    temp=0.7,
                    max_tokens=getattr(context.config, 'summary_max_tokens', 800),
                    generation_type="content_generation",
                    seed=seed,
                )

            if chunk_summary:
                chunk_results.append(chunk_summary)

        # Combine results from all chunks
        if not chunk_results:
            logger.warning("No summary was generated from any chunk")
            return ""

        summary = combine_chunk_results(chunk_results)

        logger.success(f"Generated article summary ({len(summary.split())} words)")
        return summary.strip()

    except Exception as e:
        logger.error(f"Error generating article summary: {str(e)}")
        # Return empty string on error rather than raising
        return ""
```

### Script 2: Modify `script2/article_generator/generator.py`

Find the section in the `generate_article` method that handles summary generation and replace it with:

```python
# Generate summary if enabled
if self.config.add_summary_into_article:
    try:
        provider.info("Generating article summary...")

        # Determine if we should use a separate model for summary generation
        use_separate_model = (
            hasattr(self.config, 'enable_separate_summary_model') and
            self.config.enable_separate_summary_model and
            hasattr(self.config, 'summary_keynotes_model') and
            self.config.summary_keynotes_model
        )

        # Import chunking utilities
        from article_generator.chunking_utils import chunk_article_for_processing, combine_chunk_results

        # Get the chunk size from config or use default
        chunk_size = getattr(self.config, 'summary_chunk_size', 8000)

        # Prepare article dictionary for chunking
        article_dict = {
            'title': article_components['title'],
            'introduction': article_components['introduction'],
            'sections': article_components['sections'],
            'conclusion': article_components['conclusion']
        }

        # Chunk the article if needed
        article_chunks = chunk_article_for_processing(article_dict, chunk_size=chunk_size)
        provider.info(f"Article split into {len(article_chunks)} chunks for summary generation")

        chunk_results = []

        for i, chunk in enumerate(article_chunks):
            provider.info(f"Processing chunk {i+1}/{len(article_chunks)} for summary")

            # Create a prompt for generating a summary
            summary_prompt = f"""
            Please create a concise summary of the following article about {keyword}.
            The summary should be 2-3 paragraphs long and capture the main points of the article.

            Article Title: {chunk.get('title', '')}

            Article Introduction:
            {chunk.get('introduction', '')}

            Article Content:
            {' '.join(chunk.get('sections', []))}

            Article Conclusion:
            {chunk.get('conclusion', '')}

            Summary:
            """

            # Generate summary with the appropriate model
            if use_separate_model and self.config.use_openrouter:
                provider.info(f"Using separate model for summary generation: {self.config.summary_keynotes_model}")

                # Get max tokens from config or use default
                max_tokens = getattr(self.config, 'summary_max_tokens', 800)

                # Create messages for the API call
                messages = [
                    {"role": "system", "content": "You are an expert content writer specializing in creating comprehensive article summaries."},
                    {"role": "user", "content": summary_prompt}
                ]

                # Make the API call using OpenRouter
                from utils.ai_utils import make_openrouter_api_call

                response = make_openrouter_api_call(
                    messages=messages,
                    model=self.config.summary_keynotes_model,
                    api_key=self.config.openrouter_api_key,
                    site_url=self.config.openrouter_site_url or "https://example.com",
                    site_name=self.config.openrouter_site_name or "AI Article Generator",
                    temperature=0.7,
                    max_tokens=max_tokens
                )

                chunk_summary = response.choices[0].message.content.strip()
            else:
                # Use the standard generate_completion function
                chunk_summary = generate_completion(
                    prompt=summary_prompt,
                    model=self.config.openai_model,
                    temperature=0.7,
                    max_tokens=getattr(self.config, 'summary_max_tokens', 800),
                    article_context=context
                )

            if chunk_summary:
                chunk_results.append(chunk_summary)

        # Combine results from all chunks
        if not chunk_results:
            provider.warning("No summary was generated from any chunk")
            article_components['summary'] = ""
        else:
            article_components['summary'] = combine_chunk_results(chunk_results)
            provider.success(f"Generated article summary ({len(article_components['summary'].split())} words)")

    except Exception as e:
        provider.error(f"Error generating article summary: {str(e)}")
        article_components['summary'] = ""
```

## Part 4: Keynotes Generation Enhancement

Now we need to modify the keynotes (block notes) generation functions to use the specified model and implement chunking for large articles.

### Script 1: Modify `script1/article_generator/text_processor.py`

Find the `generate_block_notes` function and replace it with:

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_block_notes(
    context: ArticleContext,
    article_content: Dict[str, str],
    blocknote_prompt: str,
    *,
    engine: str,
    enable_token_tracking: bool = False,
    track_token_usage: bool = False
) -> str:
    """
    Generate key takeaways block notes from the article content.

    Uses a large context window model if configured, with chunking for very large articles.
    """
    logger.info("Generating block notes...")

    try:
        # Determine if we should use a separate model for keynotes generation
        use_separate_model = (
            hasattr(context.config, 'enable_separate_summary_model') and
            context.config.enable_separate_summary_model and
            hasattr(context.config, 'summary_keynotes_model') and
            context.config.summary_keynotes_model
        )

        # Get the chunk size from config or use default
        chunk_size = getattr(context.config, 'keynotes_chunk_size', 8000)

        # Import chunking utilities
        from article_generator.chunking_utils import chunk_article_for_processing, combine_chunk_results

        # Chunk the article if needed
        article_chunks = chunk_article_for_processing(article_content, chunk_size=chunk_size)
        logger.info(f"Article split into {len(article_chunks)} chunks for keynotes generation")

        chunk_results = []

        for i, chunk in enumerate(article_chunks):
            logger.info(f"Processing chunk {i+1}/{len(article_chunks)} for keynotes")

            # Prepare article content for prompt
            article_text = (
                f"Title: {chunk.get('title', '')}\n\n"
                f"Introduction: {chunk.get('introduction', '')}\n\n"
            )

            # Add sections
            if isinstance(chunk.get('sections', []), list):
                article_text += "Main Content:\n"
                for section in chunk.get('sections', []):
                    article_text += f"{section}\n\n"

            # Add conclusion
            article_text += f"Conclusion: {chunk.get('conclusion', '')}"

            # Format the prompt
            prompt = blocknote_prompt.format(
                article_content=article_text,
                keyword=context.keyword
            )

            # Track token usage if enabled
            if enable_token_tracking and track_token_usage:
                prompt_tokens = len(context.encoding.encode(prompt))
                logger.debug(f"Token usage - Prompt: {prompt_tokens}")

            # Generate block notes with the appropriate model
            if use_separate_model and context.config.use_openrouter:
                logger.info(f"Using separate model for keynotes generation: {context.config.summary_keynotes_model}")

                # Use OpenRouter with the specified model
                from article_generator.content_generator import make_openrouter_api_call

                # Get max tokens from config or use default
                max_tokens = getattr(context.config, 'keynotes_max_tokens', 300)

                # Create messages for the API call
                messages = [
                    {"role": "system", "content": "You are an SEO Specialist tasked with creating a concise summary of the article's key takeaways."},
                    {"role": "user", "content": prompt}
                ]

                # Make the API call
                response = make_openrouter_api_call(
                    messages=messages,
                    model=context.config.summary_keynotes_model,
                    api_key=context.config.openrouter_api_key,
                    site_url=context.config.openrouter_site_url,
                    site_name=context.config.openrouter_site_name,
                    temperature=context.config.block_notes_temperature,
                    max_tokens=max_tokens
                )

                chunk_keynotes = response.choices[0].message.content.strip()
            else:
                # Use the standard OpenAI API
                if context.config.enable_rate_limiting and openai_rate_limiter:
                    # Define the API call function
                    def make_api_call():
                        return openai.chat.completions.create(
                            model=engine,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=context.config.block_notes_temperature,
                            max_tokens=getattr(context.config, 'keynotes_max_tokens', 300),
                            top_p=context.config.block_notes_top_p,
                            frequency_penalty=context.config.block_notes_frequency_penalty,
                            presence_penalty=context.config.block_notes_presence_penalty
                        )

                    # Execute with rate limiting
                    response = openai_rate_limiter.execute_with_rate_limit(make_api_call)
                else:
                    response = openai.chat.completions.create(
                        model=engine,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=context.config.block_notes_temperature,
                        max_tokens=getattr(context.config, 'keynotes_max_tokens', 300),
                        top_p=context.config.block_notes_top_p,
                        frequency_penalty=context.config.block_notes_frequency_penalty,
                        presence_penalty=context.config.block_notes_presence_penalty
                    )

                chunk_keynotes = response.choices[0].message.content.strip()

            # Track token usage for response if enabled
            if enable_token_tracking and track_token_usage:
                response_tokens = len(context.encoding.encode(chunk_keynotes))
                logger.debug(f"Token usage - Response: {response_tokens}")

            if chunk_keynotes:
                chunk_results.append(chunk_keynotes)

        # Combine results from all chunks
        if not chunk_results:
            logger.warning("No keynotes were generated from any chunk")
            return ""

        block_notes = combine_chunk_results(chunk_results)

        logger.success(f"Block notes generated successfully (length: {len(block_notes)})")
        return block_notes

    except Exception as e:
        logger.error(f"Error generating block notes: {str(e)}")
        return ""
```

### Script 2: Modify `script2/article_generator/text_processor.py`

Find the `generate_block_notes` function and replace it with:

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
def generate_block_notes(
    context: ArticleContext,
    article_content: Dict[str, str],
    blocknote_prompt: str,
    *,
    engine: str,
    enable_token_tracking: bool = False,
    track_token_usage: bool = False
) -> str:
    """Generate block notes (key takeaways) for an article."""
    provider.info("Generating block notes...")

    try:
        # Determine if we should use a separate model for keynotes generation
        use_separate_model = (
            hasattr(context.config, 'enable_separate_summary_model') and
            context.config.enable_separate_summary_model and
            hasattr(context.config, 'summary_keynotes_model') and
            context.config.summary_keynotes_model
        )

        # Get the chunk size from config or use default
        chunk_size = getattr(context.config, 'keynotes_chunk_size', 8000)

        # Import chunking utilities
        from article_generator.chunking_utils import chunk_article_for_processing, combine_chunk_results

        # Chunk the article if needed
        article_chunks = chunk_article_for_processing(article_content, chunk_size=chunk_size)
        provider.info(f"Article split into {len(article_chunks)} chunks for keynotes generation")

        chunk_results = []

        for i, chunk in enumerate(article_chunks):
            provider.info(f"Processing chunk {i+1}/{len(article_chunks)} for keynotes")

            # Prepare article content for prompt
            article_text = (
                f"Title: {chunk.get('title', '')}\n\n"
                f"Introduction: {chunk.get('introduction', '')}\n\n"
            )

            # Add sections
            if isinstance(chunk.get('sections', []), list):
                article_text += "Main Content:\n"
                for section in chunk.get('sections', []):
                    article_text += f"{section}\n\n"

            # Add conclusion
            article_text += f"Conclusion: {chunk.get('conclusion', '')}"

            # Format the prompt
            prompt = blocknote_prompt.format(
                article_content=article_text,
                keyword=context.keyword
            )

            # Always add request to context - ArticleContext will handle token management
            context.add_message("user", prompt)

            provider.debug(f"Sending block notes generation request to AI service")

            # Generate block notes with the appropriate model
            if use_separate_model and context.config.use_openrouter:
                provider.info(f"Using separate model for keynotes generation: {context.config.summary_keynotes_model}")

                # Use OpenRouter with the specified model
                from utils.ai_utils import make_openrouter_api_call

                # Get max tokens from config or use default
                max_tokens = getattr(context.config, 'keynotes_max_tokens', 300)

                # Create messages for the API call
                messages = [
                    {"role": "system", "content": "You are an SEO Specialist tasked with creating a concise summary of the article's key takeaways."},
                    {"role": "user", "content": prompt}
                ]

                # Make the API call
                response = make_openrouter_api_call(
                    messages=messages,
                    model=context.config.summary_keynotes_model,
                    api_key=context.config.openrouter_api_key,
                    site_url=context.config.openrouter_site_url or "https://example.com",
                    site_name=context.config.openrouter_site_name or "AI Article Generator",
                    temperature=context.config.block_notes_temperature,
                    max_tokens=max_tokens
                )

                chunk_keynotes = response.choices[0].message.content.strip()

                # Add response to context
                context.add_message("assistant", chunk_keynotes)
            else:
                # Use the unified generate_completion function that supports both OpenAI and OpenRouter
                chunk_keynotes = generate_completion(
                    prompt=prompt,
                    model=engine,
                    temperature=context.config.block_notes_temperature,
                    max_tokens=getattr(context.config, 'keynotes_max_tokens', 300),
                    article_context=context,
                    top_p=context.config.block_notes_top_p,
                    frequency_penalty=context.config.block_notes_frequency_penalty,
                    presence_penalty=context.config.block_notes_presence_penalty
                )

            if chunk_keynotes:
                chunk_results.append(chunk_keynotes)

        # Combine results from all chunks
        if not chunk_results:
            provider.warning("No keynotes were generated from any chunk")
            return ""

        block_notes = combine_chunk_results(chunk_results)

        provider.success(f"Block notes generated successfully (length: {len(block_notes)})")
        return block_notes

    except Exception as e:
        provider.error(f"Error generating block notes: {str(e)}")
        return ""
```

## Summary

This implementation plan provides a comprehensive approach to enhancing the Summarize and Keynotes sections of the article generation system. The key improvements are:

1. **Configuration Options**: Added configuration parameters to specify a separate model for summary and keynotes generation.

2. **Chunking Utilities**: Created utility functions to split large articles into manageable chunks and combine the results.

3. **Enhanced Summary Generation**: Modified the summary generation functions to use the specified model and implement chunking for large articles.

4. **Enhanced Keynotes Generation**: Modified the keynotes generation functions to use the specified model and implement chunking for large articles.

These changes will allow the system to:
- Use a large context window LLM to process the entire article
- Maintain different prompts for Summarize and Keynotes sections
- Handle very large articles through chunking when necessary

The implementation is designed to be backward compatible, so it will continue to work with the existing configuration if the new options are not specified.

