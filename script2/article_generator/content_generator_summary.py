from typing import Dict
from article_generator.article_context import ArticleContext
from article_generator.chunking_utils import chunk_article_for_processing, combine_chunk_results_with_llm
from article_generator.logger import logger
from utils.ai_utils import generate_completion, make_openrouter_api_call


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
                articleaudience=getattr(context, 'articleaudience', 'General'),
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
                    max_tokens=getattr(context.config, 'summary_max_tokens', 800),
                    seed=seed,
                    top_p=summary_top_p,
                    frequency_penalty=summary_frequency_penalty,
                    presence_penalty=summary_presence_penalty
                )

                chunk_summary = response.choices[0].message.content.strip()
            else:
                # Use the standard generate_completion function
                # Determine which model to use based on configuration
                if use_separate_model:
                    # Use the separate summary model
                    model_to_use = context.config.summary_keynotes_model
                    logger.info(f"Using separate summary model: {model_to_use}")
                else:
                    # Otherwise, determine based on whether OpenRouter is enabled
                    model_to_use = context.config.openrouter_model if (hasattr(context.config, 'use_openrouter') and context.config.use_openrouter and context.config.openrouter_api_key) else context.config.openai_model
                
                chunk_summary = generate_completion(
                    prompt=prompt,
                    model=model_to_use,
                    temperature=getattr(context.config, "content_generation_temperature", 1.0),
                    max_tokens=getattr(context.config, 'summary_max_tokens', 800),
                    article_context=context,
                    seed=seed,
                )

            if chunk_summary:
                chunk_results.append(chunk_summary)

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
