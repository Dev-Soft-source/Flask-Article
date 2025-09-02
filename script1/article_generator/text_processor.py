# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import openai
import nltk
from typing import List, Tuple, Dict, Optional
import random
import sys
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from article_generator.content_generator import ArticleContext, make_openrouter_api_call
from .logger import logger
import re
from utils.rate_limiter import openai_rate_limiter
import json
from  article_generator.wordpress_handler import get_wordpress_credentials,upload_media_to_wordpress
from utils.config import Config
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def humanize_text(
    context: ArticleContext,
    text: str,
    humanize_prompt: str,
    *,
    engine: str,
    enable_token_tracking: bool = False,
    track_token_usage: bool = False,
    content_type: str = "content"  # New parameter to identify what's being humanized
) -> str:
    """
    Make text more human-like and natural using GPT.

    Args:
        context (ArticleContext): Conversation context
        text (str): Text to humanize
        humanize_prompt (str): Template for the humanization prompt
        engine (str): OpenAI engine to use
        enable_token_tracking (bool): Whether to track token usage
        track_token_usage (bool): Whether to display token usage info
        content_type (str): Type of content being humanized (e.g., "title", "outline", "section 1")
    Returns:
        str: Humanized text
    """
    logger.info(f"Starting text humanization for {content_type}...")

    # Format the humanize prompt with the required parameters
    prompt = humanize_prompt.format(
        humanize=text  # The text to humanize is passed in the 'humanize' parameter
    )

    try:
        # Track token usage if enabled
        if enable_token_tracking:
            tokens_used = len(context.encoding.encode(prompt))
            if track_token_usage:
                logger.debug(f"Token usage - Request: {tokens_used}")

        # Check if using OpenRouter or direct OpenAI
        if context.config.use_openrouter and context.config.openrouter_api_key:
            logger.debug(f"Using OpenRouter API for {content_type} humanization")
            
            # Get OpenRouter model for humanization
            model_to_use = context.config.humanization_model if context.config.enable_separate_humanization_model else context.openrouter_model
            if context.config.openrouter_models:
                for key, full_model_id in context.config.openrouter_models.items():
                    if key.lower() in model_to_use.lower():
                        model_to_use = full_model_id
                        break
            
            response = make_openrouter_api_call(
                messages=[{"role": "user", "content": prompt}],
                model=model_to_use,
                api_key=context.config.openrouter_api_key,
                site_url=context.config.openrouter_site_url,
                site_name=context.config.openrouter_site_name,
                temperature=context.config.humanization_temperature,
                max_tokens=int(len(context.encoding.encode(text)) * 1.2),
                top_p=context.config.humanization_top_p,
                frequency_penalty=context.config.humanization_frequency_penalty,
                presence_penalty=context.config.humanization_presence_penalty
            )
        else:
            logger.debug(f"Using OpenAI API for {content_type} humanization")
            
            # Use rate limiter if available
            if context.config.enable_rate_limiting and openai_rate_limiter:
                logger.debug("Using rate limiter for OpenAI API call")

                def make_api_call():
                    return openai.chat.completions.create(
                        model=engine,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=context.config.humanization_temperature,
                        max_tokens=int(len(context.encoding.encode(text)) * 1.2),  # Allow for some expansion
                        top_p=context.config.humanization_top_p,
                        frequency_penalty=context.config.humanization_frequency_penalty,
                        presence_penalty=context.config.humanization_presence_penalty
                    )

                response = openai_rate_limiter.execute_with_rate_limit(make_api_call)
            else:
                response = openai.chat.completions.create(
                    model=engine,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=context.config.humanization_temperature,
                    max_tokens=int(len(context.encoding.encode(text)) * 1.2),  # Allow for some expansion
                    top_p=context.config.humanization_top_p,
                    frequency_penalty=context.config.humanization_frequency_penalty,
                    presence_penalty=context.config.humanization_presence_penalty
                )

        humanized_text = response.choices[0].message.content.strip()

        # Track token usage for response if enabled
        if enable_token_tracking and track_token_usage:
            response_tokens = len(context.encoding.encode(humanized_text))
            logger.debug(f"Token usage - Response: {response_tokens}")

        logger.success(f"Text humanization completed for {content_type}")
        return humanized_text

    except Exception as e:
        logger.error(f"Error in text humanization: {str(e)}")
        # Return original text if there's an error
        return text

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def check_grammar(
    context: ArticleContext,
    text: str,
    grammar_prompt: str,
    *,
    engine: str,
    enable_token_tracking: bool = False,
    track_token_usage: bool = False,
    content_type: str = "content"  # New parameter to identify what's being checked
) -> str:
    """
    Check and correct grammar in text using GPT.

    Args:
        context (ArticleContext): Conversation context
        text (str): Text to check grammar
        grammar_prompt (str): Template for the grammar check prompt
        engine (str): OpenAI engine to use
        enable_token_tracking (bool): Whether to track token usage
        track_token_usage (bool): Whether to display token usage info
        content_type (str): Type of content being checked (e.g., "title", "outline", "section 1")
    Returns:
        str: Grammar-corrected text
    """
    logger.info(f"Starting grammar check for {content_type}...")

    # Format the grammar check prompt with the required parameters
    prompt = grammar_prompt.format(
        text=text  # The text to check is passed in the 'text' parameter
    )

    try:
        # Track token usage if enabled
        if enable_token_tracking:
            tokens_used = len(context.encoding.encode(prompt))
            if track_token_usage:
                logger.debug(f"Token usage - Request: {tokens_used}")

        # Check if using OpenRouter or direct OpenAI
        if context.config.use_openrouter and context.config.openrouter_api_key:
            logger.debug(f"Using OpenRouter API for {content_type} grammar check")
            
            # Get OpenRouter model for grammar checking
            model_to_use = context.config.grammar_check_model if context.config.enable_separate_grammar_model else context.openrouter_model
            if context.config.openrouter_models:
                for key, full_model_id in context.config.openrouter_models.items():
                    if key.lower() in model_to_use.lower():
                        model_to_use = full_model_id
                        break
            
            response = make_openrouter_api_call(
                messages=[{"role": "user", "content": prompt}],
                model=model_to_use,
                api_key=context.config.openrouter_api_key,
                site_url=context.config.openrouter_site_url,
                site_name=context.config.openrouter_site_name,
                temperature=context.config.grammar_check_temperature,
                max_tokens=int(len(context.encoding.encode(text)) * 1.2),
                top_p=context.config.grammar_check_top_p,
                frequency_penalty=context.config.grammar_check_frequency_penalty,
                presence_penalty=context.config.grammar_check_presence_penalty
            )
        else:
            logger.debug(f"Using OpenAI API for {content_type} grammar check")
            
            # Use rate limiter if available
            if context.config.enable_rate_limiting and openai_rate_limiter:
                logger.debug("Using rate limiter for OpenAI API call")

                def make_api_call():
                    return openai.chat.completions.create(
                        model=engine,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=context.config.grammar_check_temperature,
                        max_tokens=int(len(context.encoding.encode(text)) * 1.2),  # Allow for some expansion
                        top_p=context.config.grammar_check_top_p,
                        frequency_penalty=context.config.grammar_check_frequency_penalty,
                        presence_penalty=context.config.grammar_check_presence_penalty
                    )

                response = openai_rate_limiter.execute_with_rate_limit(make_api_call)
            else:
                response = openai.chat.completions.create(
                    model=engine,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=context.config.grammar_check_temperature,
                    max_tokens=int(len(context.encoding.encode(text)) * 1.2),  # Allow for some expansion
                    top_p=context.config.grammar_check_top_p,
                    frequency_penalty=context.config.grammar_check_frequency_penalty,
                    presence_penalty=context.config.grammar_check_presence_penalty
                )

        corrected_text = response.choices[0].message.content.strip()

        # Track token usage for response if enabled
        if enable_token_tracking and track_token_usage:
            response_tokens = len(context.encoding.encode(corrected_text))
            logger.debug(f"Token usage - Response: {response_tokens}")

        logger.success(f"Grammar check completed for {content_type}")
        return corrected_text

    except Exception as e:
        logger.error(f"Error in grammar check: {str(e)}")
        # Return original text if there's an error
        return text

def split_text_into_sentences(text: str) -> List[str]:
    """
    Split text into individual sentences.

    Args:
        text (str): Text to split
    Returns:
        List[str]: List of sentences
    """
    try:
        logger.debug("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        sentences = nltk.sent_tokenize(text)
        logger.debug(f"Split text into {len(sentences)} sentences")
        return sentences
    except Exception as e:
        logger.error(f"Error in sentence splitting: {str(e)}")
        # Fallback to basic splitting
        logger.warning("Falling back to basic sentence splitting")
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        logger.debug(f"Basic split resulted in {len(sentences)} sentences")
        return sentences

def distribute_sentences(sentences: List[str], num_paragraphs: int) -> List[List[str]]:
    """
    Distribute sentences across paragraphs.

    Args:
        sentences (List[str]): List of sentences
        num_paragraphs (int): Number of paragraphs to create
    Returns:
        List[List[str]]: List of paragraphs (lists of sentences)
    """
    logger.debug(f"Distributing {len(sentences)} sentences into {num_paragraphs} paragraphs")

    if not sentences:
        logger.warning("No sentences to distribute")
        return []

    # Ensure num_paragraphs is valid
    original_num = num_paragraphs
    num_paragraphs = min(num_paragraphs, len(sentences))
    if original_num != num_paragraphs:
        logger.warning(f"Adjusted number of paragraphs from {original_num} to {num_paragraphs} based on sentence count")

    # Calculate base and extra sentences per paragraph
    base_sentences = len(sentences) // num_paragraphs
    extra_sentences = len(sentences) % num_paragraphs

    logger.debug(f"Base sentences per paragraph: {base_sentences}")
    if extra_sentences:
        logger.debug(f"Extra sentences to distribute: {extra_sentences}")

    paragraphs = []
    start_idx = 0

    for i in range(num_paragraphs):
        # Add an extra sentence if there are any remaining
        paragraph_size = base_sentences + (1 if i < extra_sentences else 0)
        end_idx = start_idx + paragraph_size

        paragraphs.append(sentences[start_idx:end_idx])
        logger.debug(f"Created paragraph {i+1} with {paragraph_size} sentences")
        start_idx = end_idx

    logger.success(f"Successfully distributed sentences into {len(paragraphs)} paragraphs")
    return paragraphs

def wrap_with_paragraph_tag(text: str, tag: str = 'p') -> str:
    """
    Wrap text with HTML paragraph tags and WordPress block syntax.

    Args:
        text (str): Text to wrap
        tag (str): HTML tag to use
    Returns:
        str: Wrapped text
    """
    logger.debug(f"Wrapping text with {tag} tag")
    return f'<!-- wp:{tag} --><{tag}>{text}</{tag}><!-- /wp:{tag} -->'

def create_image_block(image, align_value='center'):
    """Create a WordPress image block with proper validation."""
    img_url = image.get('wordpress_url', image.get('url', ''))
    img_id = image.get('id', '')
    img_alt = image.get('alt', '')
    img_caption = image.get('caption', '')
    alignment = image.get('alignment', 'aligncenter')
    align_value = alignment.replace('align', '') if alignment.startswith('align') else alignment
    
    if not img_url or not img_alt:
        logger.warning(f"Invalid image data: url={img_url}, alt={img_alt}")
        return ""
    
    if img_id:
        return f"""<!-- wp:image {{"id":{img_id},"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
        <figure class="wp-block-image {alignment} size-large">
            <img src="{img_url}" alt="{img_alt}" class="wp-image-{img_id}"/>
            <figcaption>{img_caption}</figcaption>
        </figure>
        <!-- /wp:image -->\n"""
    return f"""<!-- wp:image {{"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
    <figure class="wp-block-image {alignment} size-large">
        <img src="{img_url}" alt="{img_alt}" />
        <figcaption>{img_caption}</figcaption>
    </figure>
    <!-- /wp:image -->\n"""

def insert_paa_images_randomly(paragraph_text, paa_count, body_images):
    """Insert images randomly within the PAA section (placeholder implementation)."""
    if not body_images or paa_count > len(body_images):
        return [f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n']
    
    image = body_images[len(body_images) - paa_count]
    if not isinstance(image, dict) or 'url' not in image or 'alt' not in image:
        logger.warning(f"Invalid image data for random insertion: {image}")
        return [f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n']
    
    # Randomly decide whether to insert the image before or after the paragraph
    if random.choice([True, False]):
        return [
            create_image_block(image),
            f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n'
        ]
    return [
        f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n',
        create_image_block(image)
    ]

def add_paa_section(article_dict, content, body_images, config):
    """Add the People Also Ask section to the content."""
    if not article_dict.get('paa_section'):
        return

    content.append('<!-- wp:heading -->\n<h2>People Also Ask</h2>\n<!-- /wp:heading -->\n')
    paa_lines = article_dict['paa_section'].strip().split('\n')
    print(paa_lines)
    current_paragraph = []
    paa_count = 0

    for line in paa_lines:
        line = line.strip()
        paa_count += 1

        if line.lower().startswith('# people also ask') or line.lower().startswith('## people also ask'):
            continue

        if not line:
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph)
                paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)
                content.append(f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n')
                current_paragraph = []
            continue

        if re.match(r'^#+\s+', line) or (line.startswith('**') and line.endswith('**')):
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph)
                paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)
                content.append(f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n')
                current_paragraph = []

            heading_text = re.sub(r'^#+\s+', '', line)
            heading_text = re.sub(r'^\*\*(.+)\*\*$', r'\1', heading_text)
            heading_text = re.sub(r'\*([^*]+)\*', r'\1', heading_text)
            heading_text = re.sub(r'`([^`]+)`', r'\1', heading_text)
            content.append(f'<!-- wp:heading {{"level":3}} -->\n<h3>{heading_text}</h3>\n<!-- /wp:heading -->\n')

            if body_images and config.paa_image_position == "first_heading" and paa_count <= len(body_images):
                image = body_images[len(body_images) - paa_count]
                if isinstance(image, dict) and 'url' in image and 'alt' in image:
                    content.append(create_image_block(image))
        
            if body_images and config.paa_image_position == "random" and paa_count <= len(body_images):
                paragraph_text = ' '.join(current_paragraph)
                paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)
                content.extend(insert_paa_images_randomly(paragraph_text, paa_count, body_images))
            else:
                paragraph_text = ' '.join(current_paragraph)
                paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)
                content.append(f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n')

            if body_images and config.paa_image_position == "end" and paa_count <= len(body_images):
                image = body_images[len(body_images) - paa_count]
                if isinstance(image, dict) and 'url' in image and 'alt' in image:
                    content.append(create_image_block(image))
                  
        else:
            line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            line = re.sub(r'\*([^*]+)\*', r'\1', line)
            line = re.sub(r'`([^`]+)`', r'\1', line)
            current_paragraph.append(line)
            
    print(current_paragraph)
    if current_paragraph:
        paragraph_text = ' '.join(current_paragraph)
        paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
        paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
        paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)
        
        if body_images and config.paa_image_position == "first_heading" and paa_count <= len(body_images):
            image = body_images[len(body_images) - paa_count]
            if isinstance(image, dict) and 'url' in image and 'alt' in image:
                content.append(create_image_block(image))
        
        if body_images and config.paa_image_position == "random" and paa_count <= len(body_images):
            content.extend(insert_paa_images_randomly(paragraph_text, paa_count, body_images))
        else:
            content.append(f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n')

        if body_images and config.paa_image_position == "end" and paa_count <= len(body_images):
            image = body_images[len(body_images) - paa_count]
            if isinstance(image, dict) and 'url' in image and 'alt' in image:
                content.append(create_image_block(image))

def insert_images_randomly(
    para_content: str,
    section_count: int,
    body_images: Optional[List[Dict[str, str]]] = None
) -> List[str]:
    """
    Inserts an image randomly within a paragraph's sentences.

    Args:
        para_content (str): Paragraph content to process
        section_count (int): Current section index
        body_images (Optional[List[Dict[str, str]]]): List of image metadata dictionaries

    Returns:
        List[str]: Content with image inserted at a random position
    """
    content = []
    if not body_images or section_count > len(body_images):
        content.append(f'<!-- wp:paragraph -->\n<p>{para_content}</p>\n<!-- /wp:paragraph -->\n')
        return content

    image = body_images[section_count - 1]
    if not isinstance(image, dict) or 'url' not in image or 'alt' not in image:
        content.append(f'<!-- wp:paragraph -->\n<p>{para_content}</p>\n<!-- /wp:paragraph -->\n')
        return content

    img_url = image.get('wordpress_url', image.get('url', ''))
    img_id = image.get('id', '')
    img_alt = image.get('alt', '')
    img_caption = image.get('caption', '')
    alignment = image.get('alignment', 'aligncenter')
    align_value = alignment.replace('align', '')

    image_block = f'''<!-- wp:image {{"id":{img_id},"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
    <figure class="wp-block-image {alignment} size-large">
        <img src="{img_url}" alt="{img_alt}" class="wp-image-{img_id}"/>
        <figcaption>{img_caption}</figcaption>
    </figure>
    <!-- /wp:image -->\n''' if img_id else f'''<!-- wp:image {{"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
    <figure class="wp-block-image {alignment} size-large">
        <img src="{img_url}" alt="{img_alt}" />
        <figcaption>{img_caption}</figcaption>
    </figure>
    <!-- /wp:image -->\n'''

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para_content.strip()) if s.strip()]
    if len(sentences) <= 1:
        content.append(f'<!-- wp:paragraph -->\n<p>{para_content}</p>\n<!-- /wp:paragraph -->\n')
        content.append(image_block)
        return content

    insert_index = random.randint(0, len(sentences) - 2)
    current_content = ""
    for i, sentence in enumerate(sentences):
        current_content += sentence + (". " if i < len(sentences) - 1 else "")
        if i == insert_index:
            if current_content:
                content.append(f'<!-- wp:paragraph -->\n<p>{current_content.rstrip()}</p>\n<!-- /wp:paragraph -->\n')
            content.append(image_block)
            current_content = ""

    if current_content:
        content.append(f'<!-- wp:paragraph -->\n<p>{current_content.rstrip()}</p>\n<!-- /wp:paragraph -->\n')

    return content

def insert_keytakeaway_randomly(
    para_content: str,
    config: Config,
    article_dict: Dict[str, str]
) -> List[str]:
    """
    Inserts key takeaways block at a random position between sentences in the paragraph.

    Args:
        para_content (str): Paragraph content to process
        config (Config): Configuration object
        article_dict (Dict[str, str]): Dictionary containing article data including block notes

    Returns:
        List[str]: Content with key takeaways inserted at a random position
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para_content.strip()) if s.strip()]
    result = []

    if config.keytakeaways_position == "random" and article_dict.get('block_notes'):
        clean_notes = clean_markdown_formatting(article_dict['block_notes'].replace('## Key Takeaways\n\n', '').strip())
        key_takeaways_block = [
            '<!-- wp:heading -->\n<h2>Key Takeaways</h2>\n<!-- /wp:heading -->\n',
            '<!-- wp:quote {"className":"key-takeaways-block"} -->',
            '<blockquote class="wp-block-quote key-takeaways-block">',
        ]
        paragraphs = [p.strip() for p in clean_notes.split('\n\n') if p.strip()]
        key_takeaways_block.extend(f'<p>{p}</p>' for p in paragraphs)
        key_takeaways_block.extend(['</blockquote>', '<!-- /wp:quote -->'])

        if len(sentences) > 2:
            insert_position = random.randint(1, len(sentences) - 1)
            current_content = ""
            for i, sentence in enumerate(sentences):
                current_content += sentence + (". " if i < len(sentences) - 1 else "")
                if i == insert_position - 1:
                    result.append(f'<!-- wp:paragraph -->\n<p>{current_content.rstrip()}</p>\n<!-- /wp:paragraph -->\n')
                    result.extend(key_takeaways_block)
                    current_content = ""
            if current_content:
                result.append(f'<!-- wp:paragraph -->\n<p>{current_content.rstrip()}</p>\n<!-- /wp:paragraph -->\n')
        else:
            result.append(f'<!-- wp:paragraph -->\n<p>{para_content}</p>\n<!-- /wp:paragraph -->\n')
            if key_takeaways_block:  # Check if key_takeaways_block is non-empty
                result.extend(key_takeaways_block)
    else:
        result.append(f'<!-- wp:paragraph -->\n<p>{para_content}</p>\n<!-- /wp:paragraph -->\n')
        if article_dict.get('block_notes'):  # Only extend if block_notes exist
            clean_notes = clean_markdown_formatting(article_dict['block_notes'].replace('## Key Takeaways\n\n', '').strip())
            key_takeaways_block = [
                '<!-- wp:heading -->\n<h2>Key Takeaways</h2>\n<!-- /wp:heading -->\n',
                '<!-- wp:quote {"className":"key-takeaways-block"} -->',
                '<blockquote class="wp-block-quote key-takeaways-block">',
            ]
            paragraphs = [p.strip() for p in clean_notes.split('\n\n') if p.strip()]
            key_takeaways_block.extend(f'<p>{p}</p>' for p in paragraphs)
            key_takeaways_block.extend(['</blockquote>', '<!-- /wp:quote -->'])
            result.extend(key_takeaways_block)

    return result


def generate_random_index(lst: List[str]) -> int:
    """
    Generate a random index from a list.

    Args:
        lst (List[str]): List to generate index from

    Returns:
        int: Random index or -1 if list is empty
    """
    return random.randint(0, len(lst) - 1) if lst else -1


def check_and_add_key_takeaways(content_list: List[str], article_dict: Dict[str, str]) -> List[str]:
    """
    Checks if key takeaways heading exists and adds it in the middle if not present.

    Args:
        content_list (List[str]): List of content strings
        article_dict (Dict[str, str]): Dictionary containing article data including block notes

    Returns:
        List[str]: Content list with key takeaways added if needed
    """
    pattern = r'<!-- wp:heading -->\n<h2>Key Takeaways</h2>\n<!-- /wp:heading -->\n'
    if any(re.search(pattern, item, re.DOTALL) for item in content_list) or not article_dict.get('block_notes'):
        return content_list.copy()

    result = content_list.copy()
    middle_index = len(result) // 4 if result else 0
    clean_notes = clean_markdown_formatting(article_dict['block_notes'].replace('## Key Takeaways\n\n', '').strip())
    key_takeaways_block = [
        '<!-- wp:heading -->\n<h2>Key Takeaways</h2>\n<!-- /wp:heading -->\n',
        '<!-- wp:quote {"className":"key-takeaways-block"} -->',
        '<blockquote class="wp-block-quote key-takeaways-block">',
    ]
    paragraphs = [p.strip() for p in clean_notes.split('\n\n') if p.strip()]
    key_takeaways_block.extend(f'<p>{p}</p>' for p in paragraphs)
    key_takeaways_block.extend(['</blockquote>', '<!-- /wp:quote -->'])

    result.insert(middle_index, '\n'.join(key_takeaways_block))
    return result


def insert_youtube_randomly() -> str:
    """
    Randomly picks a position for YouTube video insertion.

    Returns:
        str: Random position identifier ('rand1', 'rand2', or 'rand3')
    """
    positions = ["rand1", "rand2", "rand3"]
    return random.choice(positions)


def format_article_for_wordpress(
    config: Config,
    article_dict: Dict[str, str],
    youtube_position: str = "after_introduction",
    body_images: Optional[List[Dict[str, str]]] = None,
    add_summary: bool = False
) -> str:
    """
    Format article content for WordPress using Gutenberg blocks and proper HTML tags.

    Args:
        config (Config): Configuration object
        article_dict (Dict[str, str]): Article components
        youtube_position (str): Where to place YouTube video ("after_introduction", "after_first_section", "end", "random")
        body_images (List[Dict[str, str]], optional): List of body images with their metadata
        add_summary (bool): Whether to add a summary section at the start of the article

    Returns:
        str: Formatted article content for WordPress
    """
    logger.info("Formatting article for WordPress...")
    with open('article_dict.json', 'w') as f:
        json.dump(article_dict, f, indent=4)

    logger.debug(f"Article dict keys: {list(article_dict.keys())}")
    if 'external_links' in article_dict:
        logger.debug(f"External links found (length: {len(article_dict['external_links'])})")
        if article_dict['external_links']:
            logger.debug(f"First 100 chars of external links: {article_dict['external_links'][:100]}...")
    else:
        logger.warning("'external_links' key not found")

    content = []
    place_random_picked = insert_youtube_randomly() if youtube_position == "random" else None

    # 1. Add summary if enabled
    if add_summary and article_dict.get('summary'):
        content.append('<!-- wp:heading {"style":{"typography":{"fontStyle":"normal","fontWeight":"700"},"spacing":{"margin":{"bottom":"1.5em"}}}} -->')
        content.append('<h2>Summary</h2>')
        content.append('<!-- /wp:heading -->')
        summary_text = clean_markdown_formatting(article_dict["summary"])
        content.append('<!-- wp:paragraph {"className":"article-summary"} -->')
        content.append(f'<p class="article-summary">{summary_text}</p>')
        content.append('<!-- /wp:paragraph -->')
        content.append('<!-- wp:separator {"className":"is-style-wide","style":{"spacing":{"margin":{"top":"2em","bottom":"2em"}}}} -->')
        content.append('<hr class="wp-block-separator has-alpha-channel-opacity is-style-wide"/>')
        content.append('<!-- /wp:separator -->')

    # 2. Add introduction
    if article_dict.get('introduction'):
        paragraphs = article_dict['introduction'].split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                content.append(f'<!-- wp:paragraph -->\n<p>{paragraph.strip()}</p>\n<!-- /wp:paragraph -->\n')

    # Add YouTube video after introduction
    if youtube_position == "first_heading" and article_dict.get('youtube_video'):
        content.append(f'<!-- wp:paragraph -->\n{article_dict["youtube_video"]}\n<!-- /wp:paragraph -->\n')
    if place_random_picked == "rand1" and article_dict.get('youtube_video'):
        content.append(f'<!-- wp:paragraph -->\n{article_dict["youtube_video"]}\n<!-- /wp:paragraph -->\n')

    # 3. Add sections with hierarchical support
    num_of_sections = len(article_dict.get('sections', []))
   
    section_to_add_keytakeaway = generate_random_index(article_dict.get('sections'))
    added = False

    section_count = 0
    for section in article_dict.get('sections', []):
        if section.strip():
            lines = section.split('\n')
            heading = lines[0].lstrip('#').strip() if lines[0].startswith('#') else lines[0].strip()
            content.append(f'<!-- wp:heading -->\n<h2>{heading}</h2>\n<!-- /wp:heading -->\n')

            if body_images and section_count <= len(body_images) and config.image_position == "under_first_heading":
                image = body_images[section_count - 1]
                if isinstance(image, dict) and 'url' in image and 'alt' in image:
                    img_url = image.get('wordpress_url', image.get('url', ''))
                    img_id = image.get('id', '')
                    img_alt = image.get('alt', '')
                    img_caption = image.get('caption', '')
                    alignment = image.get('alignment', 'aligncenter')
                    align_value = alignment.replace('align', '')
                    image_block = f'''<!-- wp:image {{"id":{img_id},"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                    <figure class="wp-block-image {alignment} size-large">
                        <img src="{img_url}" alt="{img_alt}" class="wp-image-{img_id}"/>
                        <figcaption>{img_caption}</figcaption>
                    </figure>
                    <!-- /wp:image -->\n''' if img_id else f'''<!-- wp:image {{"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                    <figure class="wp-block-image {alignment} size-large">
                        <img src="{img_url}" alt="{img_alt}" />
                        <figcaption>{img_caption}</figcaption>
                    </figure>
                    <!-- /wp:image -->\n'''
                    content.append(image_block)

            section_content = '\n'.join(lines[1:]).strip()
            
            # Handle hierarchical structure: check for h3 and h4 headings
            h3_pattern = r'<h3>(.*?)</h3>\s*<p>(.*?)</p>'
            h4_pattern = r'<h4>(.*?)</h4>\s*<p>(.*?)</p>'
            
            h3_matches = re.findall(h3_pattern, section_content, re.DOTALL)
            h4_matches = re.findall(h4_pattern, section_content, re.DOTALL)
            
            if h3_matches or h4_matches:
                # Handle 3-level hierarchy (h2 -> h3 -> h4)
                if h3_matches:
                    for h3_heading, h3_content in h3_matches:
                        content.append(f'<!-- wp:heading {"level":3} -->\n<h3>{h3_heading}</h3>\n<!-- /wp:heading -->\n')
                        
                        # Check for h4 subsections within h3 content
                        h4_in_h3 = re.findall(h4_pattern, h3_content, re.DOTALL)
                        if h4_in_h3:
                            for h4_heading, h4_content in h4_in_h3:
                                content.append(f'<!-- wp:heading {"level":4} -->\n<h4>{h4_heading}</h4>\n<!-- /wp:heading -->\n')
                                content.append(f'<!-- wp:paragraph -->\n<p>{h4_content.strip()}</p>\n<!-- /wp:paragraph -->\n')
                        else:
                            content.append(f'<!-- wp:paragraph -->\n<p>{h3_content.strip()}</p>\n<!-- /wp:paragraph -->\n')
                
                # Handle 2-level hierarchy (h2 -> h4)
                elif h4_matches:
                    for h4_heading, h4_content in h4_matches:
                        content.append(f'<!-- wp:heading {"level":4} -->\n<h4>{h4_heading}</h4>\n<!-- /wp:heading -->\n')
                        content.append(f'<!-- wp:paragraph -->\n<p>{h4_content.strip()}</p>\n<!-- /wp:paragraph -->\n')
                        
            else:
                # Handle flat structure (no subheadings)
                paragraphs = section_content.split('\n\n')
                para_count = 0
                for paragraph in paragraphs:
                    para_count += 1
                    if paragraph.strip():
                        if re.match(r'^#+\s+', paragraph.strip()):
                            subheading = paragraph.strip().lstrip('#').strip()
                            level = paragraph.count('#', 0, paragraph.find(' '))
                            level = min(level + 1, 6)  # Ensure we don't exceed h6
                            content.append(f'<!-- wp:heading {"level":{level}} -->\n<h{level}>{subheading}</h{level}>\n<!-- /wp:heading -->\n')
                        else:
                            content.append(f'<!-- wp:paragraph -->\n<p>{paragraph.strip()}</p>\n<!-- /wp:paragraph -->\n')

            put_here = num_of_sections - 1
            if num_of_sections > 1 and put_here == section_count and config.keytakeaways_position == "middle" and article_dict.get('block_notes'):
                content.append('<!-- wp:heading -->\n<h2>Key Takeaways</h2>\n<!-- /wp:heading -->\n')
                clean_notes = clean_markdown_formatting(article_dict['block_notes'].replace('## Key Takeaways\n\n', '').strip())
                content.append('<!-- wp:quote {"className":"key-takeaways-block"} -->')
                content.append('<blockquote class="wp-block-quote key-takeaways-block">')
                paragraphs = clean_notes.split('\n\n')
                for paragraph in paragraphs:
                    if paragraph.strip():
                        content.append(f'<p>{paragraph.strip()}</p>')
                content.append('</blockquote>')
                content.append('<!-- /wp:quote -->')
                
            section_count += 1

    if num_of_sections <= 1 and config.keytakeaways_position == "middle" and article_dict.get('block_notes'):
        content.append('<!-- wp:heading -->\n<h2>Key Takeaways</h2>\n<!-- /wp:heading -->\n')
        clean_notes = clean_markdown_formatting(article_dict['block_notes'].replace('## Key Takeaways\n\n', '').strip())
        content.append('<!-- wp:quote {"className":"key-takeaways-block"} -->')
        content.append('<blockquote class="wp-block-quote key-takeaways-block">')
        paragraphs = clean_notes.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                content.append(f'<p>{paragraph.strip()}</p>')
        content.append('</blockquote>')
        content.append('<!-- /wp:quote -->')

    if place_random_picked == "rand3" and article_dict.get('youtube_video'):
        content.append(f'<!-- wp:paragraph -->\n{article_dict["youtube_video"]}\n<!-- /wp:paragraph -->\n')

    # 4. Add PAA section
    add_paa_section(article_dict, content, body_images, config)
    
    # 5. Add key takeaways before conclusion
    if config.keytakeaways_position == "before_conclusion" and article_dict.get('block_notes'):
        content.append('<!-- wp:heading -->\n<h2>Key Takeaways</h2>\n<!-- /wp:heading -->\n')
        clean_notes = clean_markdown_formatting(article_dict['block_notes'].replace('## Key Takeaways\n\n', '').strip())
        content.append('<!-- wp:quote {"className":"key-takeaways-block"} -->')
        content.append('<blockquote class="wp-block-quote key-takeaways-block">')
        paragraphs = clean_notes.split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                content.append(f'<p>{paragraph.strip()}</p>')
        content.append('</blockquote>')
        content.append('<!-- /wp:quote -->')

    # 6. Add conclusion
    if article_dict.get('conclusion'):
        content.append('<!-- wp:heading -->\n<h2>Conclusion</h2>\n<!-- /wp:heading -->\n')
        paragraphs = article_dict['conclusion'].split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                content.append(f'<!-- wp:paragraph -->\n<p>{paragraph.strip()}</p>\n<!-- /wp:paragraph -->\n')

    # Add YouTube video at end
    if youtube_position == "end" and article_dict.get('youtube_video'):
        content.append(f'<!-- wp:paragraph -->\n{article_dict["youtube_video"]}\n<!-- /wp:paragraph -->\n')
    if place_random_picked == "rand2" and article_dict.get('youtube_video'):
        content.append(f'<!-- wp:paragraph -->\n{article_dict["youtube_video"]}\n<!-- /wp:paragraph -->\n')

    # 7. Add FAQ section
    if article_dict.get('faq_section'):
        content.append('<!-- wp:heading -->\n<h2>Frequently Asked Questions</h2>\n<!-- /wp:heading -->\n')
        faq_lines = article_dict['faq_section'].strip().split('\n')
        current_paragraph = []
        is_question = True

        for line in faq_lines:
            line = line.strip()
            if line.lower().startswith('# frequently asked questions') or line.lower().startswith('## frequently asked questions'):
                continue

            if not line:
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
                    paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
                    paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)
                    content.append(f'<!-- wp:heading {"level":3} -->\n<h3>{paragraph_text}</h3>\n<!-- /wp:heading -->\n' if is_question else f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n')
                    current_paragraph = []
                    is_question = not is_question
                continue

            if re.match(r'^#+\s+', line) or (line.startswith('**') and line.endswith('**')) or line.startswith('Q:'):
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
                    paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
                    paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)
                    content.append(f'<!-- wp:heading {"level":3} -->\n<h3>{paragraph_text}</h3>\n<!-- /wp:heading -->\n' if is_question else f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n')
                    current_paragraph = []

                heading_text = line.replace('Q:', '').strip() if line.startswith('Q:') else line
                heading_text = re.sub(r'^#+\s+', '', heading_text)
                heading_text = re.sub(r'^\*\*(.+)\*\*$', r'\1', heading_text)
                heading_text = re.sub(r'\*([^*]+)\*', r'\1', heading_text)
                heading_text = re.sub(r'`([^`]+)`', r'\1', heading_text)
                current_paragraph = [heading_text]
                is_question = True
            elif line.startswith('A:'):
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
                    paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
                    paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)
                    content.append(f'<!-- wp:heading {"level":3} -->\n<h3>{paragraph_text}</h3>\n<!-- /wp:heading -->\n' if is_question else f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n')
                    current_paragraph = []

                answer_text = line.replace('A:', '').strip()
                answer_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer_text)
                answer_text = re.sub(r'\*([^*]+)\*', r'\1', answer_text)
                answer_text = re.sub(r'`([^`]+)`', r'\1', answer_text)
                current_paragraph = [answer_text]
                is_question = False
            else:
                line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
                line = re.sub(r'\*([^*]+)\*', r'\1', line)
                line = re.sub(r'`([^`]+)`', r'\1', line)
                current_paragraph.append(line)

        if current_paragraph:
            paragraph_text = ' '.join(current_paragraph)
            paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
            paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
            paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)
            content.append(f'<!-- wp:heading {"level":3} -->\n<h3>{paragraph_text}</h3>\n<!-- /wp:heading -->\n' if is_question else f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n')

    # 8. Add external links
    if article_dict.get('external_links') and article_dict['external_links'].strip():
        logger.debug(f"Adding external links (length: {len(article_dict['external_links'])})")
        logger.debug(f"External links start: {article_dict['external_links'][:50]}...")
        if '<!-- wp:' in article_dict['external_links']:
            logger.debug("External links already in WordPress format")
            content.append(article_dict['external_links'])
        else:
            logger.debug("Formatting external links")
            content.append('<!-- wp:heading -->\n<h2>External Resources</h2>\n<!-- /wp:heading -->\n')
            content.append('<!-- wp:paragraph -->\n<p>Here are some helpful resources for more information about this topic:</p>\n<!-- /wp:paragraph -->\n')
            lines = article_dict['external_links'].strip().split('\n')
            for line in lines:
                if line.strip().startswith('- ['):
                    match = re.match(r'- \[(.*?)\]\((.*?)\)(.*)', line.strip())
                    if match:
                        title, url, snippet = match.groups()
                        content.append('<!-- wp:paragraph -->')
                        content.append(f'<p><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a><br>{snippet.strip()}</p>')
                        content.append('<!-- /wp:paragraph -->\n')
                elif line.strip():
                    content.append(f'<!-- wp:paragraph -->\n<p>{line.strip()}</p>\n<!-- /wp:paragraph -->\n')
    else:
        logger.warning(f"External links {'empty' if 'external_links' in article_dict else 'not found'}")

    # 9. Add random key takeaways
    if config.keytakeaways_position == "random" and article_dict.get('block_notes'):
        content = check_and_add_key_takeaways(content, article_dict).copy()

    result = '\n'.join(content)
    logger.debug(f"Total content length: {len(result)}")
    logger.debug(f"External links heading in content: {'External Resources' in result}")
    logger.info(result)

    return result


def clean_markdown_formatting(text: str) -> str:
    """
    Clean Markdown formatting from text for WordPress display.

    Args:
        text (str): Text with potential Markdown formatting
    Returns:
        str: Cleaned text with Markdown formatting removed
    """
    if not text:
        return ""

    # Remove bold formatting (**text**)
    cleaned_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)

    # Remove italic formatting (*text*)
    cleaned_text = re.sub(r'\*([^*]+)\*', r'\1', cleaned_text)

    # Remove code ticks (`text`)
    cleaned_text = re.sub(r'`([^`]+)`', r'\1', cleaned_text)

    # Remove heading hashtags (# Heading)
    cleaned_text = re.sub(r'^#+\s*', '', cleaned_text, flags=re.MULTILINE)

    return cleaned_text.strip()

def convert_wp_to_markdown(content: str) -> str:
    """
    Convert WordPress content to Markdown format.

    Args:
        content (str): WordPress formatted content
    Returns:
        str: Markdown formatted content
    """
    logger.info("Converting WordPress content to Markdown...")
    logger.debug(f"Input content length: {len(content)}")

    # Remove WordPress block comments and tags
    content = re.sub(r'<!-- wp:paragraph -->\s*<p>', '', content)
    content = re.sub(r'</p>\s*<!-- /wp:paragraph -->', '', content)
    content = re.sub(r'<!-- wp:heading -->\s*<h2>', '## ', content)  # Ensure space after ##
    content = re.sub(r'</h2>\s*<!-- /wp:heading -->', '', content)
    content = re.sub(r'<!-- wp:heading {"level":3} -->\s*<h3>', '### ', content)  # Ensure space after ###
    content = re.sub(r'</h3>\s*<!-- /wp:heading -->', '', content)
    content = re.sub(r'<!-- wp:heading {"level":4} -->\s*<h4>', '#### ', content)  # Ensure space after ####
    content = re.sub(r'</h4>\s*<!-- /wp:heading -->', '', content)
    content = re.sub(r'<!-- wp:heading {"level":5} -->\s*<h5>', '##### ', content)  # Ensure space after #####
    content = re.sub(r'</h5>\s*<!-- /wp:heading -->', '', content)
    content = re.sub(r'<!-- wp:heading {"level":6} -->\s*<h6>', '###### ', content)  # Ensure space after ######
    content = re.sub(r'</h6>\s*<!-- /wp:heading -->', '', content)

    # Convert bold/strong tags to <b> tags
    content = re.sub(r'<strong>(.*?)</strong>', r'<b>\1</b>', content)

    logger.success(f"Conversion complete (output length: {len(content)})")
    return content

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_block_notes(
    context: ArticleContext,
    article_content: Dict[str, str],
    blocknote_prompt: str,
    combine_prompt: str,
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
        from article_generator.chunking_utils import chunk_article_for_processing, combine_chunk_results, combine_chunk_results_with_llm

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
                keyword=context.keyword,
                articleaudience=context.config.articleaudience,
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

        # Use the LLM to combine chunks if there are multiple chunks
        if len(chunk_results) > 1:
            logger.info("Using LLM to combine keynotes chunks")
            block_notes = combine_chunk_results_with_llm(
                results=chunk_results,
                context=context,
                combine_prompt=combine_prompt,
                is_summary=False
            )
        else:
            block_notes = chunk_results[0]

        logger.success(f"Block notes generated successfully (length: {len(block_notes)})")
        return block_notes

    except Exception as e:
        logger.error(f"Error generating block notes: {str(e)}")
        return ""

def parse_paragraph_with_heading(content: str) -> Tuple[str, str]:
    """
    Parse content to extract paragraph heading and content.
    
    Args:
        content (str): HTML content with potential heading and paragraph tags
        
    Returns:
        Tuple[str, str]: A tuple of (heading, paragraph_content)
    """
    if not content:
        return ("", "")
        
    # Parse the response to extract heading and content
    pattern = r'<h4>(.*?)</h4>\s*<p>(.*?)</p>'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        heading = match.group(1).strip()
        paragraph = match.group(2).strip()
        return heading, paragraph
    
    # Also try to parse raw LLM output format if HTML tags are not found
    content_match = re.search(r'<paragraph>(.*?)</paragraph>', content, re.DOTALL)
    heading_match = re.search(r'<heading>(.*?)</heading>', content, re.DOTALL)
    
    if content_match and heading_match:
        paragraph = content_match.group(1).strip()
        heading = heading_match.group(1).strip()
        return heading, paragraph
    
    # Legacy support for old square bracket format
    content_match = re.search(r'\[CONTENT\](.*?)(?=\[HEADING\])', content, re.DOTALL)
    heading_match = re.search(r'\[HEADING\](.*)', content, re.DOTALL)
    
    if content_match and heading_match:
        paragraph = content_match.group(1).strip()
        heading = heading_match.group(1).strip()
        return heading, paragraph
    
    # Fallback if the format is not correct
    return "Additional Information", content.strip()

def convert_to_markdown(content: str) -> str:
    """
    Convert content with HTML paragraph headings to Markdown.
    
    Args:
        content (str): HTML formatted content with potential heading and paragraph tags
        
    Returns:
        str: Markdown formatted content
    """
    logger.info("Converting content to Markdown...")
    
    # Parse content looking for paragraph headings (<h4>) and paragraphs (<p>)
    pattern = r'<h4>(.*?)</h4>\s*<p>(.*?)</p>'
    matches = re.findall(pattern, content, re.DOTALL)
    
    if matches:
        markdown_content = []
        for para_heading, para_content in matches:
            # Clean HTML tags from paragraph content first
            para_content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', para_content)
            para_content = re.sub(r'<em>(.*?)</em>', r'*\1*', para_content)
            para_content = re.sub(r'<[^>]+>', '', para_content)  # Remove any other HTML tags
            
            # Add content first, then heading (to match the new approach)
            markdown_content.append(para_content)
            markdown_content.append("")  # Empty line for proper Markdown formatting
            markdown_content.append(f"#### {para_heading}")
            markdown_content.append("")  # Empty line between paragraph blocks
        
        return "\n".join(markdown_content)
    else:
        # Fallback to simpler conversion if no heading/paragraph pairs found
        # First convert headings
        content = re.sub(r'<h2>(.*?)</h2>', r'## \1\n', content)
        content = re.sub(r'<h3>(.*?)</h3>', r'### \1\n', content)
        content = re.sub(r'<h4>(.*?)</h4>', r'#### \1\n', content)
        
        # Convert paragraphs
        content = re.sub(r'<p>(.*?)</p>', r'\1\n\n', content)
        
        # Convert formatting tags
        content = re.sub(r'<strong>(.*?)</strong>', r'**\1**', content)
        content = re.sub(r'<em>(.*?)</em>', r'*\1*', content)
        
        # Remove any remaining HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        return content.strip()