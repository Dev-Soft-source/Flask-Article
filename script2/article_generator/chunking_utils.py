# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from typing import Dict, List, Any
from utils.rich_provider import provider


def chunk_article_for_processing(
    article_dict: Dict[str, Any], chunk_size: int = 8000, overlap: int = 500
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
    # Convert chunk_size and overlap to integers in case they're passed as floats
    chunk_size = int(chunk_size)
    overlap = int(overlap)
    provider.info(
        f"Chunking article for processing (chunk_size={chunk_size}, overlap={overlap})"
    )

    # If the article is small enough, return it as is
    full_content = (
        f"Title: {article_dict.get('title', '')}\n\n"
        f"Introduction: {article_dict.get('introduction', '')}\n\n"
    )

    # Add sections
    if isinstance(article_dict.get("sections", []), list):
        for i, section in enumerate(article_dict.get("sections", [])):
            full_content += f"Section {i+1}: {section}\n\n"

    # Add conclusion
    full_content += f"Conclusion: {article_dict.get('conclusion', '')}"

    # If content is small enough, return as single chunk
    if len(full_content) <= chunk_size:
        provider.info("Article is small enough to process in one chunk")
        return [article_dict]

    # Otherwise, split into chunks
    provider.info(
        f"Article is too large ({len(full_content)} chars), splitting into chunks"
    )

    chunks = []
    current_chunk = {
        "title": article_dict.get("title", ""),
        "introduction": "",
        "sections": [],
        "conclusion": "",
    }
    current_size = len(article_dict.get("title", ""))

    # Always include the title in each chunk

    # Add introduction if it fits
    intro = article_dict.get("introduction", "")
    if current_size + len(intro) <= chunk_size:
        current_chunk["introduction"] = intro
        current_size += len(intro)
    else:
        # Split introduction if needed
        current_chunk["introduction"] = intro[: chunk_size - current_size]
        provider.warning("Introduction had to be split across chunks")

    # Add sections
    if isinstance(article_dict.get("sections", []), list):
        for section in article_dict.get("sections", []):
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
                        "title": article_dict.get("title", ""),
                        "introduction": "",
                        "sections": overlap_sections,
                        "conclusion": "",
                    }
                    current_size = len(article_dict.get("title", "")) + sum(
                        len(s) for s in overlap_sections
                    )

                # Now add the current section
                if len(section) <= chunk_size:
                    current_chunk["sections"].append(section)
                    current_size += len(section)
                else:
                    # If the section itself is too large, split it
                    provider.warning(
                        f"Section is too large ({len(section)} chars), splitting it"
                    )
                    section_parts = []
                    # Ensure we're using integers for the range function
                    step = int(chunk_size - overlap)
                    for i in range(0, len(section), step):
                        part = section[i : i + step]
                        section_parts.append(part)

                    # Add first part to current chunk
                    current_chunk["sections"].append(section_parts[0])
                    current_size += len(section_parts[0])

                    # Create new chunks for remaining parts
                    for part in section_parts[1:]:
                        chunks.append(current_chunk)
                        current_chunk = {
                            "title": article_dict.get("title", ""),
                            "introduction": "",
                            "sections": [part],
                            "conclusion": "",
                        }
                        current_size = len(article_dict.get("title", "")) + len(part)

    # Add conclusion to the last chunk if it fits
    conclusion = article_dict.get("conclusion", "")
    if current_size + len(conclusion) <= chunk_size:
        current_chunk["conclusion"] = conclusion
    else:
        # Add current chunk and create a final one for conclusion
        chunks.append(current_chunk)
        current_chunk = {
            "title": article_dict.get("title", ""),
            "introduction": "",
            "sections": [],
            "conclusion": conclusion,
        }

    # Add the final chunk
    chunks.append(current_chunk)

    provider.success(f"Split article into {len(chunks)} chunks")
    return chunks


def combine_chunk_results(results: List[str]) -> str:
    """
    Combine results from processing multiple chunks using a simple approach.

    Args:
        results: List of text results from processing each chunk

    Returns:
        Combined result
    """
    provider.info(f"Combining results from {len(results)} chunks using simple approach")

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
            sentences = result.split(". ")
            new_content = []

            for sentence in sentences:
                # Only add sentences that aren't substantially contained in the combined result
                if (
                    sentence
                    and sentence.strip()
                    and sentence.strip() + "." not in combined
                ):
                    new_content.append(sentence)

            if new_content:
                combined += " " + ". ".join(new_content)
                if not combined.endswith("."):
                    combined += "."

    provider.success(f"Combined chunk results (length: {len(combined)})")
    return combined


def combine_chunk_results_with_llm(
    results: List[str], context, combine_prompt: str, is_summary: bool = True
) -> str:
    """
    Combine results from processing multiple chunks using the LLM itself.

    Args:
        results: List of text results from processing each chunk
        context: The ArticleContext object containing configuration
        combine_prompt: The prompt template to use for combining chunks
        is_summary: Whether this is for summary (True) or keynotes (False)

    Returns:
        Combined result that is coherent and non-repetitive
    """
    provider.info(f"Combining results from {len(results)} chunks using LLM")

    if len(results) == 1:
        return results[0]

    try:
        # Determine if we should use a separate model
        use_separate_model = (
            hasattr(context.config, "enable_separate_summary_model")
            and context.config.enable_separate_summary_model
            and hasattr(context.config, "summary_keynotes_model")
            and context.config.summary_keynotes_model
        )

        # Format the chunks for the LLM
        chunks_text = ""
        for i, result in enumerate(results):
            chunks_text += f"CHUNK {i+1}:\n{result}\n\n"

        # Get the number of paragraphs to generate based on content type
        if is_summary:
            num_paragraphs = getattr(
                context.config, "summary_combination_paragraphs", 2
            )
        else:
            num_paragraphs = getattr(
                context.config, "keynotes_combination_paragraphs", 1
            )

        # Create the prompt for the LLM
        content_type = "summary" if is_summary else "key takeaways"

        # Format the prompt with required parameters
        if is_summary:
            # For summaries, pass both num_paragraphs and chunks_text
            prompt = combine_prompt.format(
                num_paragraphs=num_paragraphs, chunks_text=chunks_text
            )
        else:
            # For blocknotes, pass both num_paragraphs and chunks_text
            prompt = combine_prompt.format(
                num_paragraphs=num_paragraphs, chunks_text=chunks_text
            )

        # Add the combination prompt to the context for debugging
        provider.info(f"Adding chunk combination prompt to context")
        context.add_message(
            "user",
            f"Please combine these {content_type} chunks into a single coherent result:\n\n{chunks_text}",
        )

        # Get the appropriate max tokens
        if is_summary:
            # First try to get from token_limits dictionary, then fall back to summary_max_tokens
            max_tokens = context.config.token_limits.get(
                "summary_combination",
                getattr(context.config, "summary_max_tokens", 800),
            )
        else:
            # First try to get from token_limits dictionary, then fall back to keynotes_max_tokens
            max_tokens = context.config.token_limits.get(
                "keynotes_combination",
                getattr(context.config, "keynotes_max_tokens", 300),
            )

        # Use the appropriate model to combine the chunks
        if use_separate_model and context.config.use_openrouter:
            provider.info(
                f"Using separate model for combining chunks: {context.config.summary_keynotes_model}"
            )

            # Import the OpenRouter API call function
            from utils.ai_utils import make_openrouter_api_call

            # Create messages for the API call
            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert at combining multiple article {content_type} chunks into a single coherent result.",
                },
                {"role": "user", "content": prompt},
            ]

            # Get generation parameters from config or use defaults based on content type
            if is_summary:
                combine_temperature = getattr(
                    context.config, "summary_combination_temperature", 0.3
                )
            else:
                combine_temperature = getattr(
                    context.config, "keynotes_combination_temperature", 0.3
                )

            combine_top_p = getattr(context.config, "combine_top_p", 1.0)
            combine_frequency_penalty = getattr(
                context.config, "combine_frequency_penalty", 0.0
            )
            combine_presence_penalty = getattr(
                context.config, "combine_presence_penalty", 0.0
            )

            # Get seed if seed control is enabled
            combine_seed = (
                getattr(context.config, "combine_seed", None)
                if context.config.enable_seed_control
                else None
            )

            # Make the API call with all parameters
            response = make_openrouter_api_call(
                messages=messages,
                model=context.config.summary_keynotes_model,
                api_key=context.config.openrouter_api_key,
                site_url=context.config.openrouter_site_url,
                site_name=context.config.openrouter_site_name,
                temperature=combine_temperature,
                max_tokens=max_tokens,
                seed=combine_seed,
                top_p=combine_top_p,
                frequency_penalty=combine_frequency_penalty,
                presence_penalty=combine_presence_penalty,
            )

            combined = response.choices[0].message.content.strip()

            # Add the combined response to the context for debugging
            provider.info(f"Adding combined {content_type} response to context")
            context.add_message("assistant", combined)
        else:
            # Use the standard OpenAI API
            import openai
            from utils.rate_limiter import openai_rate_limiter

            # Determine the engine to use
            if use_separate_model:
                # Use the separate model for summary/keynotes if configured
                engine = context.config.summary_keynotes_model
                provider.info(f"Using separate summary/keynotes model for combining chunks: {engine}")
            else:
                # Default to the configured model based on OpenRouter settings
                engine = context.config.openrouter_model if (hasattr(context.config, 'use_openrouter') and context.config.use_openrouter and context.config.openrouter_api_key) else context.config.openai_model
                provider.info(f"Using standard model for combining chunks: {engine}")

            # Get the temperature for combining chunks based on content type
            if is_summary:
                temperature = getattr(
                    context.config, "summary_combination_temperature", 0.3
                )
            else:
                temperature = getattr(
                    context.config, "keynotes_combination_temperature", 0.3
                )

            if context.config.enable_rate_limiting and openai_rate_limiter:
                # Define the API call function
                def make_api_call():
                    return openai.chat.completions.create(
                        model=engine,
                        messages=[
                            {
                                "role": "system",
                                "content": f"You are an expert at combining multiple article {content_type} chunks into a single coherent result.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                # Execute with rate limiting
                response = openai_rate_limiter.execute_with_rate_limit(make_api_call)
            else:
                # Create parameters dictionary so we can conditionally add seed if seed control is enabled
                params = {
                    "model": engine,
                    "messages": [
                        {
                            "role": "system",
                            "content": f"You are an expert at combining multiple article {content_type} chunks into a single coherent result.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                
                # Add seed if seed control is enabled
                if context.config.enable_seed_control:
                    combine_seed = getattr(context.config, "combine_seed", None)
                    if combine_seed is not None:
                        params["seed"] = combine_seed
                
                response = openai.chat.completions.create(**params)

            combined = response.choices[0].message.content.strip()

            # Add the combined response to the context for debugging
            provider.info(f"Adding combined {content_type} response to context")
            context.add_message("assistant", combined)

        provider.success(
            f"Successfully combined chunks using LLM (length: {len(combined)})"
        )
        return combined

    except Exception as e:
        provider.error(f"Error combining chunks with LLM: {str(e)}")
        provider.warning("Falling back to simple chunk combination method")
        # Fall back to the simple method if the LLM approach fails
        return combine_chunk_results(results)
