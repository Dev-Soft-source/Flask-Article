# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import openai
import nltk
from typing import List, Tuple, Dict, Optional
import random
import sys
import os
import time
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from article_generator.content_generator import ArticleContext
from utils.rich_provider import provider
from utils.ai_utils import generate_completion, make_openrouter_api_call
import re
from openai import APIError, RateLimitError
from config import Config

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Rate limiting configuration
MAX_RETRIES = 5
INITIAL_DELAY = 1  # Initial delay in seconds
MAX_DELAY = 60     # Maximum delay in seconds

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
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
    provider.info(f"Starting text humanization for {content_type}...")

    # Format the humanize prompt with the required parameters
    prompt = humanize_prompt.format(
        humanize=text  # The text to humanize is passed in the 'humanize' parameter
    )

    try:
        # Track token usage if enabled
        if enable_token_tracking and track_token_usage:
            tokens_used = context.count_message_tokens({"role": "user", "content": prompt})
            provider.token_usage(f"Humanization for {content_type} - Request tokens: {tokens_used}")

        # Always add request to context - ArticleContext will handle token management
        context.add_message("user", prompt)

        # Determine which model to use for humanization
        model_to_use = context.config.humanization_model if context.config.enable_separate_humanization_model else engine
        provider.debug(f"Sending request to AI service for {content_type} humanization using model: {model_to_use}")

        # Use the unified generate_completion function that supports both OpenAI and OpenRouter
        humanized_text = generate_completion(
            prompt=prompt,
            model=model_to_use,
            temperature=context.config.humanization_temperature,
            max_tokens=len(text) + 100,
            article_context=context,
            top_p=context.config.humanization_top_p,
            frequency_penalty=context.config.humanization_frequency_penalty,
            presence_penalty=context.config.humanization_presence_penalty
        )

        provider.success(f"Text humanization complete for {content_type} (length: {len(humanized_text)})")
        return humanized_text

    except Exception as e:
        provider.error(f"Error in humanizing {content_type}: {str(e)}")
        return text

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
        text (str): Text to check
        grammar_prompt (str): Template for grammar checking prompt
        engine (str): OpenAI engine to use
        enable_token_tracking (bool): Whether to track token usage
        track_token_usage (bool): Whether to display token usage info
        content_type (str): Type of content being checked (e.g., "title", "outline", "section 1")
    Returns:
        str: Corrected text
    """
    provider.info(f"Starting grammar check for {content_type}...")

    # Format the prompt with the text
    formatted_prompt = grammar_prompt.format(text=text)

    # Track token usage if enabled
    if enable_token_tracking and track_token_usage:
        tokens_used = context.count_message_tokens({"role": "user", "content": formatted_prompt})
        provider.token_usage(f"Grammar check for {content_type} - Request tokens: {tokens_used}")

    # Maximum retry attempts with longer delays for rate limit errors
    max_retries = 3
    rate_limit_delay = 20  # Longer delay for rate limit errors (in seconds)
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Always add request to context - ArticleContext will handle token management
            context.add_message("user", formatted_prompt)

            provider.debug(f"Sending grammar check request to AI service for {content_type} (length: {len(text)})")

            # Use the unified generate_completion function that supports both OpenAI and OpenRouter
            corrected_text = generate_completion(
                prompt=formatted_prompt,
                model=context.config.grammar_check_model if context.config.enable_separate_grammar_model else engine,
                temperature=context.config.grammar_check_temperature,
                max_tokens=len(text) + 50,
                article_context=context,
                top_p=context.config.grammar_check_top_p,
                frequency_penalty=context.config.grammar_check_frequency_penalty,
                presence_penalty=context.config.grammar_check_presence_penalty
            )

            provider.success(f"Grammar check complete for {content_type} (length: {len(corrected_text)})")
            return corrected_text

        except RateLimitError as e:
            retry_count += 1
            if retry_count >= max_retries:
                provider.warning(f"Rate limit exceeded for grammar check after {max_retries} attempts. Returning original text.")
                return text
                
            provider.warning(f"Rate limit error for grammar check ({retry_count}/{max_retries}): {str(e)}. Waiting {rate_limit_delay}s before retry...")
            time.sleep(rate_limit_delay)
            # Increase delay for next retry
            rate_limit_delay *= 2
            
        except requests.exceptions.HTTPError as e:
            # Check if it's a 429 error (rate limit)
            if "429" in str(e):
                retry_count += 1
                if retry_count >= max_retries:
                    provider.warning(f"Rate limit exceeded (HTTP 429) for grammar check after {max_retries} attempts. Returning original text.")
                    return text
                    
                provider.warning(f"HTTP 429 Rate limit error for grammar check ({retry_count}/{max_retries}): {str(e)}. Waiting {rate_limit_delay}s before retry...")
                time.sleep(rate_limit_delay)
                # Increase delay for next retry
                rate_limit_delay *= 2
            else:
                # For other HTTP errors, log and return original text
                provider.error(f"HTTP error checking grammar for {content_type}: {str(e)}")
                return text
                
        except Exception as e:
            provider.error(f"Error checking grammar for {content_type}: {str(e)}")
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
        provider.debug("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        sentences = nltk.sent_tokenize(text)
        provider.debug(f"Split text into {len(sentences)} sentences")
        return sentences
    except Exception as e:
        provider.error(f"Error in sentence splitting: {str(e)}")
        # Fallback to basic splitting
        provider.warning("Falling back to basic sentence splitting")
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        provider.debug(f"Basic split resulted in {len(sentences)} sentences")
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
    provider.debug(f"Distributing {len(sentences)} sentences into {num_paragraphs} paragraphs")

    if not sentences:
        provider.warning("No sentences to distribute")
        return []

    # Ensure num_paragraphs is valid
    original_num = num_paragraphs
    num_paragraphs = min(num_paragraphs, len(sentences))
    if original_num != num_paragraphs:
        provider.warning(f"Adjusted number of paragraphs from {original_num} to {num_paragraphs} based on sentence count")

    # Calculate base and extra sentences per paragraph
    base_sentences = len(sentences) // num_paragraphs
    extra_sentences = len(sentences) % num_paragraphs

    provider.debug(f"Base sentences per paragraph: {base_sentences}")
    if extra_sentences:
        provider.debug(f"Extra sentences to distribute: {extra_sentences}")

    paragraphs = []
    start_idx = 0

    for i in range(num_paragraphs):
        # Add an extra sentence if there are any remaining
        paragraph_size = base_sentences + (1 if i < extra_sentences else 0)
        end_idx = start_idx + paragraph_size

        paragraphs.append(sentences[start_idx:end_idx])
        provider.debug(f"Created paragraph {i+1} with {paragraph_size} sentences")
        start_idx = end_idx

    provider.success(f"Successfully distributed sentences into {len(paragraphs)} paragraphs")
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
    provider.debug(f"Wrapping text with {tag} tag")
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
        provider.warning(f"Invalid image data: url={img_url}, alt={img_alt}")
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

def insert_paa_images_randomly(paragraph_text, paa_count, image):
    """Insert images randomly within the PAA section."""
    
    if not isinstance(image, dict) or 'url' not in image or 'alt' not in image:
        provider.warning(f"Invalid image data for random insertion: {image}")
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


def add_paa_section(article_dict, content, body_images, config, start):
    """Add the People Also Ask section with one image per section, preventing duplicates."""
    if not article_dict.get('paa_section'):
        return

    # Initialize sets to track unique images and text content
    used_image_urls = set()
    processed_items = set()
    
    image_count = max(0, min(start, len(body_images) - 1))  # Clamp start
    content.append('<!-- wp:heading -->\n<h2>People Also Ask</h2>\n<!-- /wp:heading -->\n')
    paa_lines = article_dict['paa_section'].strip().split('\n')
    current_paragraph = []
    section_count = 0  # Track headings for image insertion
    section_has_image = False  # Track if current section has an image
    count = 0
    for line in paa_lines:
        
        line = line.strip()

        if line.lower().startswith('# people also ask') or line.lower().startswith('## people also ask'):
            continue

        # Handle empty lines to finalize paragraphs
        if not line:
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph).strip()
                paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)

                if paragraph_text and paragraph_text not in processed_items:
                    processed_items.add(paragraph_text)
                    # if config.paa_image_position != "random":
                    content.append(f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n')
                   

                    # Add image for 'random' or 'end' only if no image in section yet
                    if (body_images and not section_has_image and image_count < len(body_images) and (config.paa_image_position in ["middle"])):
                        image = body_images[image_count]
                        if isinstance(image, dict) and 'url' in image and 'alt' in image and image['url'] not in used_image_urls:
                            if config.paa_image_position == "random":
                                content.extend(insert_paa_images_randomly(paragraph_text, section_count, image))
                            else:  # end
                                content.append(create_image_block(image))
                            used_image_urls.add(image['url'])
                            section_has_image = True
                            image_count += 1
                        else:
                            print(f"Invalid or duplicate image at index {image_count}: {image}")
                current_paragraph = []
            continue

        # Process headings (questions) - handle different markdown heading formats
        if re.match(r'^#+\s+', line) or (line.startswith('**') and line.endswith('**')):
            # Finalize any existing paragraph before new heading
            if current_paragraph:
                paragraph_text = ' '.join(current_paragraph).strip()
                paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
                paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)

                if paragraph_text and paragraph_text not in processed_items:
                    processed_items.add(paragraph_text)
                    # if config.paa_image_position != "random":
                    content.append(f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n')
                       
                    # Add image for 'random' or 'end' only if no image in section yet
                    if (body_images and not section_has_image and image_count < len(body_images) and (config.paa_image_position in ["middle"])):
                        image = body_images[image_count]
                        if isinstance(image, dict) and 'url' in image and 'alt' in image and image['url'] not in used_image_urls:
                            if config.paa_image_position == "random":
                                content.extend(insert_paa_images_randomly(paragraph_text, section_count, image))
                            else:  # end
                                content.append(create_image_block(image))
                            used_image_urls.add(image['url'])
                            section_has_image = True
                            image_count += 1
                        else:
                            print(f"Invalid or duplicate image at index {image_count}: {image}")
                current_paragraph = []

            # Clean and format the heading
            heading_text = line
            heading_text = re.sub(r'^#+\s+', '', heading_text)
            heading_text = re.sub(r'^\*\*(.+)\*\*$', r'\1', heading_text)
            heading_text = re.sub(r'\*([^*]+)\*', r'\1', heading_text)
            heading_text = re.sub(r'`([^`]+)`', r'\1', heading_text)

            if heading_text not in processed_items:
                processed_items.add(heading_text)
                content.append(f'<!-- wp:heading {"level":3} -->\n<h3>{heading_text}</h3>\n<!-- /wp:heading -->\n')
                section_count += 1
                section_has_image = False  # Reset image flag for new section

                # Add image for 'first_heading' configuration
                if body_images and config.paa_image_position == "first_heading" and image_count < len(body_images):
                    image = body_images[image_count]
                    if isinstance(image, dict) and 'url' in image and 'alt' in image and image['url'] not in used_image_urls:
                        content.append(create_image_block(image))
                        used_image_urls.add(image['url'])
                        section_has_image = True
                        image_count += 1
                    else:
                        print(f"Invalid or duplicate image at index {image_count}: {image}")

        else:
            # Collect regular paragraph content
            line = re.sub(r'\*\*([^*]+)\*\*', r'\1', line)
            line = re.sub(r'\*([^*]+)\*', r'\1', line)
            line = re.sub(r'`([^`]+)`', r'\1', line)
            current_paragraph.append(line)

    # Add any remaining paragraph content
    if current_paragraph:
        paragraph_text = ' '.join(current_paragraph).strip()
        paragraph_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', paragraph_text)
        paragraph_text = re.sub(r'\*([^*]+)\*', r'\1', paragraph_text)
        paragraph_text = re.sub(r'`([^`]+)`', r'\1', paragraph_text)

        if paragraph_text and paragraph_text not in processed_items:
            processed_items.add(paragraph_text)
            # if config.paa_image_position != "random":
            content.append(f'<!-- wp:paragraph -->\n<p>{paragraph_text}</p>\n<!-- /wp:paragraph -->\n')
            
            # if config.paa_image_position == "random" and paa_lines[]

            # Add image for 'random' or 'end' only if no image in section yet
            if (body_images and not section_has_image and image_count < len(body_images) and
                    (config.paa_image_position in ["middle"])):
                image = body_images[image_count]
                if isinstance(image, dict) and 'url' in image and 'alt' in image and image['url'] not in used_image_urls:
                    if config.paa_image_position == "random":
                        content.extend(insert_paa_images_randomly(paragraph_text, section_count, image))
                    else:  # end
                        content.append(create_image_block(image))
                    used_image_urls.add(image['url'])
                    section_has_image = True
                    image_count += 1
                else:
                    print(f"Invalid or duplicate image at index {image_count}: {image}")
                    


def generate_random_index(lst: List[str]) -> int:
    """
    Generate a random index from a list.

    Args:
        lst (List[str]): List to generate index from

    Returns:
        int: Random index or -1 if list is empty
    """
    return random.randint(0, len(lst) - 1) if lst else -1


def insert_youtube_randomly() -> str:
    """
    Randomly picks a position for YouTube video insertion.

    Returns:
        str: Random position identifier ('rand1', 'rand2', or 'rand3')
    """
    positions = ["rand1", "rand2", "rand3"]
    return positions[random.randint(0, len(positions) - 1)]


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
    if not body_images or section_count >= len(body_images):
        content.append(f'<!-- wp:paragraph -->\n<p>{para_content}</p>\n<!-- /wp:paragraph -->\n')
        return content

    image = body_images[section_count]
    if not isinstance(image, dict) or 'url' not in image:
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

    sentences = [s.strip() for s in para_content.split('. ') if s.strip()]
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
            result.extend(key_takeaways_block)
    else:
        result.append(f'<!-- wp:paragraph -->\n<p>{para_content}</p>\n<!-- /wp:paragraph -->\n')

    return result


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



def format_article_for_wordpress(
    config: Config,
    article_dict: Dict[str, str],
    body_images: Optional[List[Dict[str, str]]] = None,
    add_summary: bool = False,
    add_block_notes: bool = True
) -> str:
    """
    Format article content for WordPress using Gutenberg blocks and proper HTML tags.

    Args:
        config (Config): Configuration object
        article_dict (Dict[str, str]): Article components
        body_images (List[Dict[str, str]], optional): List of body images with their metadata
        add_summary (bool): Whether to add a summary section at the start of the article
        add_block_notes (bool): Whether to add block notes (key takeaways) section

    Returns:
        str: Formatted article content for WordPress
    """
    provider.info("Formatting article for WordPress...")
    content = []
    

    if config.youtube_position == "random":
        place_random_picked = insert_youtube_randomly()

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

    # Add YouTube video after introduction if configured
    if config.youtube_position == "first_heading" and article_dict.get('youtube_video'):
        content.append(f'<!-- wp:paragraph -->\n{article_dict["youtube_video"]}\n<!-- /wp:paragraph -->\n')
    
    # Add YouTube video to rand1 if configured
    if place_random_picked == "rand1" and article_dict.get('youtube_video'):
        content.append(f'<!-- wp:paragraph -->\n{article_dict["youtube_video"]}\n<!-- /wp:paragraph -->\n')

    num_of_sections = len(article_dict.get('sections', []))
    added = False
    countImages = 0
    # 3. Process main sections with headings and images
    if 'headings' in article_dict and 'sections' in article_dict:
        section_to_add_keytakeaway = generate_random_index(article_dict.get('sections'))
        
        for i, (heading, section) in enumerate(zip(article_dict['headings'], article_dict['sections'])):
            # Add section heading
            if heading:
                content.append(f'<!-- wp:heading -->\n<h2>{heading}</h2>\n<!-- /wp:heading -->\n')

            # Add body image at the beginning of the section if available
            if body_images and i < len(body_images) and config.image_position == "under_first heading":
                image = body_images[countImages]
                countImages += 1
                if isinstance(image, dict) and 'url' in image:
                    img_url = image.get('wordpress_url', image.get('url', ''))
                    img_id = image.get('id', '')
                    img_alt = image.get('alt', '')
                    img_caption = image.get('caption', '')
                    alignment = image.get('alignment', 'aligncenter')
                    align_value = alignment.replace('align', '')

                    if img_id:
                        content.append(f'''<!-- wp:image {{"id":{img_id},"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                        <figure class="wp-block-image {alignment} size-large">
                            <img src="{img_url}" alt="{img_alt}" class="wp-image-{img_id}"/>
                            <figcaption>{img_caption}</figcaption>
                        </figure>
                        <!-- /wp:image -->\n''')
                    else:
                        content.append(f'''<!-- wp:image {{"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                        <figure class="wp-block-image {alignment} size-large">
                            <img src="{img_url}" alt="{img_alt}" />
                            <figcaption>{img_caption}</figcaption>
                        </figure>
                        <!-- /wp:image -->\n''')

            # Process section content
            if section:
                pattern = r'<h4>(.*?)</h4>\s*<p>(.*?)</p>'
                matches = re.findall(pattern, section, re.DOTALL)
                
                if matches:
                    para_count = 0
                    for para_heading, para_content in matches:
                        para_count += 1
                        content.append(f'<!-- wp:heading {{"level":4}} -->\n<h4>{para_heading}</h4>\n<!-- /wp:heading -->\n')
                        
                        new_content_pattern = []
                        if body_images and i < len(body_images) and config.image_position == "middle":
                            image = body_images[countImages]
                            countImages += 1
                            if isinstance(image, dict) and 'url' in image:
                                img_url = image.get('wordpress_url', image.get('url', ''))
                                img_id = image.get('id', '')
                                img_alt = image.get('alt', '')
                                img_caption = image.get('caption', '')
                                alignment = image.get('alignment', 'aligncenter')
                                align_value = alignment.replace('align', '')

                                if img_id:
                                    new_content_pattern.append(f'''<!-- wp:image {{"id":{img_id},"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                                    <figure class="wp-block-image {alignment} size-large">
                                        <img src="{img_url}" alt="{img_alt}" class="wp-image-{img_id}"/>
                                        <figcaption>{img_caption}</figcaption>
                                    </figure>
                                    <!-- /wp:image -->\n''')
                                else:
                                    new_content_pattern.append(f'''<!-- wp:image {{"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                                    <figure class="wp-block-image {alignment} size-large">
                                        <img src="{img_url}" alt="{img_alt}" />
                                        <figcaption>{img_caption}</figcaption>
                                    </figure>
                                    <!-- /wp:image -->\n''')

                        if body_images and i <= len(body_images) and config.image_position not in ["middle", "random"]:
                            content.append(f'<!-- wp:paragraph -->\n<p>{para_content}</p>\n<!-- /wp:paragraph -->\n')

                        if new_content_pattern:
                            sentences = re.findall(r'[^.!?]+[.!?]?', para_content.strip())
                            mid_point = len(sentences) // 2
                            first_half = ' '.join(sentences[:mid_point]).strip()
                            second_half = ' '.join(sentences[mid_point:]).strip()

                            if first_half:
                                content.append(f'<!-- wp:paragraph -->\n<p>{first_half}</p>\n<!-- /wp:paragraph -->\n')
                            content.extend(new_content_pattern)
                            if second_half:
                                content.append(f'<!-- wp:paragraph -->\n<p>{second_half}</p>\n<!-- /wp:paragraph -->\n')

                        if config.image_position == "random":
                            content.extend(insert_images_randomly(para_content, countImages, body_images=body_images))
                            countImages += 1

                        if config.keytakeaways_position == "random" and added == False and i == section_to_add_keytakeaway and article_dict.get('block_notes'):
                            content.extend(insert_keytakeaway_randomly(para_content, config, article_dict))
                            added = True
                               # Add body image at the beginning of the section if available
                               
                        if body_images and i < len(body_images) and config.image_position == "end":
                            image = body_images[countImages]
                            countImages += 1
                            if isinstance(image, dict) and 'url' in image:
                                img_url = image.get('wordpress_url', image.get('url', ''))
                                img_id = image.get('id', '')
                                img_alt = image.get('alt', '')
                                img_caption = image.get('caption', '')
                                alignment = image.get('alignment', 'aligncenter')
                                align_value = alignment.replace('align', '')

                                if img_id:
                                    content.append(f'''<!-- wp:image {{"id":{img_id},"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                                    <figure class="wp-block-image {alignment} size-large">
                                        <img src="{img_url}" alt="{img_alt}" class="wp-image-{img_id}"/>
                                        <figcaption>{img_caption}</figcaption>
                                    </figure>
                                    <!-- /wp:image -->\n''')
                                else:
                                    content.append(f'''<!-- wp:image {{"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                                    <figure class="wp-block-image {alignment} size-large">
                                        <img src="{img_url}" alt="{img_alt}" />
                                        <figcaption>{img_caption}</figcaption>
                                    </figure>
                                    <!-- /wp:image -->\n''')

                else:
                    paragraphs = section.split('\n\n')
                  
                    para_count = 0
                    for paragraph in paragraphs:
                        if paragraph.strip():
                            if paragraph.startswith('###'):
                                subheading = paragraph.replace('###', '').strip()
                                content.append(f'<!-- wp:heading {{"level":3}} -->\n<h3>{subheading}</h3>\n<!-- /wp:heading -->\n')
                            else:
                                if body_images and i <= len(body_images) and config.image_position not in ["middle", "random"]:
                                    content.append(f'<!-- wp:paragraph -->\n<p>{paragraph.strip()}</p>\n<!-- /wp:paragraph -->\n')
                                
                                new_content_pattern = []
                                if body_images and i < len(body_images) and config.image_position == "middle":
                                    image = body_images[countImages]
                                    countImages += 1
                                    if isinstance(image, dict) and 'url' in image:
                                        img_url = image.get('wordpress_url', image.get('url', ''))
                                        img_id = image.get('id', '')
                                        img_alt = image.get('alt', '')
                                        img_caption = image.get('caption', '')
                                        alignment = image.get('alignment', 'aligncenter')
                                        align_value = alignment.replace('align', '')

                                        if img_id:
                                            new_content_pattern.append(f'''<!-- wp:image {{"id":{img_id},"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                                            <figure class="wp-block-image {alignment} size-large">
                                                <img src="{img_url}" alt="{img_alt}" class="wp-image-{img_id}"/>
                                                <figcaption>{img_caption}</figcaption>
                                            </figure>
                                            <!-- /wp:image -->\n''')
                                        else:
                                            new_content_pattern.append(f'''<!-- wp:image {{"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                                            <figure class="wp-block-image {alignment} size-large">
                                                <img src="{img_url}" alt="{img_alt}" />
                                                <figcaption>{img_caption}</figcaption>
                                            </figure>
                                            <!-- /wp:image -->\n''')

                                if new_content_pattern:
                                    sentences = re.findall(r'[^.!?]+[.!?]?', paragraph.strip())
                                    mid_point = len(sentences) // 2
                                    first_half = ' '.join(sentences[:mid_point]).strip()
                                    second_half = ' '.join(sentences[mid_point:]).strip()

                                    if first_half:
                                        content.append(f'<!-- wp:paragraph -->\n<p>{first_half}</p>\n<!-- /wp:paragraph -->\n')
                                    content.extend(new_content_pattern)
                                    if second_half:
                                        content.append(f'<!-- wp:paragraph -->\n<p>{second_half}</p>\n<!-- /wp:paragraph -->\n')

                                if config.image_position == "random":
                                    content.extend(insert_images_randomly(paragraph, countImages, body_images=body_images))
                                    countImages += 1


                                if config.keytakeaways_position == "random" and added == False and article_dict.get('block_notes') and i == section_to_add_keytakeaway and article_dict.get('block_notes'):
                                    content.extend(insert_keytakeaway_randomly(paragraph, config, article_dict))
                                    added = True
                               
                                para_count += 1

                                # Add body image at the end of the section
                                if body_images and i < len(body_images) and config.image_position == "end":
                                    image = body_images[countImages]
                                    countImages += 1
                                    if isinstance(image, dict) and 'url' in image:
                                        img_url = image.get('wordpress_url', image.get('url', ''))
                                        img_id = image.get('id', '')
                                        img_alt = image.get('alt', '')
                                        img_caption = image.get('caption', '')
                                        alignment = image.get('alignment', 'aligncenter')
                                        align_value = alignment.replace('align', '')

                                        if img_id:
                                            content.append(f'''<!-- wp:image {{"id":{img_id},"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                                            <figure class="wp-block-image {alignment} size-large">
                                                <img src="{img_url}" alt="{img_alt}" class="wp-image-{img_id}"/>
                                                <figcaption>{img_caption}</figcaption>
                                            </figure>
                                            <!-- /wp:image -->\n''')
                                        else:
                                            content.append(f'''<!-- wp:image {{"sizeSlug":"large","linkDestination":"none","align":"{align_value}"}} -->
                                            <figure class="wp-block-image {alignment} size-large">
                                                <img src="{img_url}" alt="{img_alt}" />
                                                <figcaption>{img_caption}</figcaption>
                                            </figure>
                                            <!-- /wp:image -->\n''')

            # Add block notes in the middle
            put_here = num_of_sections - 1
            if num_of_sections > 1 and put_here == i and config.keytakeaways_position == "middle" and article_dict.get('block_notes'):
                provider.info("Adding block notes (Key Takeaways) to WordPress content")
                content.append('<!-- wp:heading -->\n<h2>Key Takeaways</h2>\n<!-- /wp:heading -->\n')
                notes = article_dict['block_notes'].replace('## Key Takeaways\n\n', '').strip()
                clean_notes = clean_markdown_formatting(notes)
                content.append('<!-- wp:quote {"className":"key-takeaways-block"} -->')
                content.append('<blockquote class="wp-block-quote key-takeaways-block">')
                paragraphs = clean_notes.split('\n\n')
                if len(paragraphs) > 1:
                    for paragraph in paragraphs:
                        if paragraph.strip():
                            content.append(f'<p>{paragraph.strip()}</p>')
                else:
                    content.append(f'<p>{clean_notes}</p>')
                content.append('</blockquote>')
                content.append('<!-- /wp:quote -->')

    # 4. Add block notes for single or no sections
    if num_of_sections <= 1 and article_dict.get('block_notes') and config.keytakeaways_position == "middle":
        provider.info("Adding block notes (Key Takeaways) to WordPress content")
        content.append('<!-- wp:heading -->\n<h2>Key Takeaways</h2>\n<!-- /wp:heading -->\n')
        notes = article_dict['block_notes'].replace('## Key Takeaways\n\n', '').strip()
        clean_notes = clean_markdown_formatting(notes)
        content.append('<!-- wp:quote {"className":"key-takeaways-block"} -->')
        content.append('<blockquote class="wp-block-quote key-takeaways-block">')
        paragraphs = clean_notes.split('\n\n')
        if len(paragraphs) > 1:
            for paragraph in paragraphs:
                if paragraph.strip():
                    content.append(f'<p>{paragraph.strip()}</p>')
        else:
            content.append(f'<p>{clean_notes}</p>')
        content.append('</blockquote>')
        content.append('<!-- /wp:quote -->')

    # Add YouTube video at rand3
    if place_random_picked == "rand3" and article_dict.get('youtube_video'):
        content.append(f'<!-- wp:paragraph -->\n{article_dict["youtube_video"]}\n<!-- /wp:paragraph -->\n')

    # 5. Add PAA section
    add_paa_section(article_dict, content, body_images, config, countImages)
   

    # 6. Add block notes before conclusion
    if article_dict.get('block_notes') and config.keytakeaways_position == "before_conclusion":
        provider.info("Adding block notes (Key Takeaways) to WordPress content")
        content.append('<!-- wp:heading -->\n<h2>Key Takeaways</h2>\n<!-- /wp:heading -->\n')
        notes = article_dict['block_notes'].replace('## Key Takeaways\n\n', '').strip()
        clean_notes = clean_markdown_formatting(notes)
        content.append('<!-- wp:quote {"className":"key-takeaways-block"} -->')
        content.append('<blockquote class="wp-block-quote key-takeaways-block">')
        paragraphs = clean_notes.split('\n\n')
        if len(paragraphs) > 1:
            for paragraph in paragraphs:
                if paragraph.strip():
                    content.append(f'<p>{paragraph.strip()}</p>')
        else:
            content.append(f'<p>{clean_notes}</p>')
        content.append('</blockquote>')
        content.append('<!-- /wp:quote -->')

    # 7. Add conclusion
    if article_dict.get('conclusion'):
        content.append('<!-- wp:heading -->\n<h2>Conclusion</h2>\n<!-- /wp:heading -->\n')
        paragraphs = article_dict['conclusion'].split('\n\n')
        for paragraph in paragraphs:
            if paragraph.strip():
                content.append(f'<!-- wp:paragraph -->\n<p>{paragraph.strip()}</p>\n<!-- /wp:paragraph -->\n')

    # Add YouTube video at end
    if config.youtube_position == "end" and article_dict.get('youtube_video'):
        content.append(f'<!-- wp:paragraph -->\n{article_dict["youtube_video"]}\n<!-- /wp:paragraph -->\n')

    # Add YouTube video at rand2
    if place_random_picked == "rand2" and article_dict.get('youtube_video'):
        content.append(f'<!-- wp:paragraph -->\n{article_dict["youtube_video"]}\n<!-- /wp:paragraph -->\n')

    # 8. Add FAQ section
    if article_dict.get('faq_section'):
        content.append('<!-- wp:heading -->\n<h2>Frequently Asked Questions</h2>\n<!-- /wp:heading -->\n')
        faq_items = article_dict['faq_section'].split('\n\n')

        for item in faq_items:
            if not item.strip():
                continue

            clean_item = item.strip()
            if clean_item.lower().startswith('# frequently asked questions') or clean_item.lower().startswith('## frequently asked questions'):
                continue

            if clean_item.startswith('Q:'):
                question = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_item.replace('Q:', '').strip())
                question = re.sub(r'\*([^*]+)\*', r'\1', question)
                question = re.sub(r'^#+\s*', '', question)
                question = re.sub(r'`([^`]+)`', r'\1', question)
                content.append(f'<!-- wp:heading {{"level":3}} -->\n<h3>{question}</h3>\n<!-- /wp:heading -->\n')

            elif clean_item.startswith('A:'):
                answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_item.replace('A:', '').strip())
                answer = re.sub(r'\*([^*]+)\*', r'\1', answer)
                answer = re.sub(r'^#+\s*', '', answer)
                answer = re.sub(r'`([^`]+)`', r'\1', answer)
                content.append(f'<!-- wp:paragraph -->\n<p>{answer}</p>\n<!-- /wp:paragraph -->\n')

            elif '**' in clean_item and clean_item.startswith('**'):
                match = re.match(r'\*\*([^*]+)\*\*\s*(.*)', clean_item)
                if match:
                    question = re.sub(r'^#+\s*', '', match.group(1).strip())
                    question = re.sub(r'`([^`]+)`', r'\1', question)
                    content.append(f'<!-- wp:heading {{"level":3}} -->\n<h3>{question}</h3>\n<!-- /wp:heading -->\n')
                    if match.group(2).strip():
                        remaining = re.sub(r'\*\*([^*]+)\*\*', r'\1', match.group(2).strip())
                        remaining = re.sub(r'\*([^*]+)\*', r'\1', remaining)
                        remaining = re.sub(r'^#+\s*', '', remaining)
                        remaining = re.sub(r'`([^`]+)`', r'\1', remaining)
                        content.append(f'<!-- wp:paragraph -->\n<p>{remaining}</p>\n<!-- /wp:paragraph -->\n')
                else:
                    clean_item = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_item)
                    clean_item = re.sub(r'\*([^*]+)\*', r'\1', clean_item)
                    clean_item = re.sub(r'^#+\s*', '', clean_item)
                    clean_item = re.sub(r'`([^`]+)`', r'\1', clean_item)
                    content.append(f'<!-- wp:paragraph -->\n<p>{clean_item}</p>\n<!-- /wp:paragraph -->\n')

            else:
                clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_item)
                clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)
                clean_text = re.sub(r'^#+\s*', '', clean_text)
                clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)
                if clean_text.endswith('?'):
                    content.append(f'<!-- wp:heading {{"level":3}} -->\n<h3>{clean_text}</h3>\n<!-- /wp:heading -->\n')
                else:
                    content.append(f'<!-- wp:paragraph -->\n<p>{clean_text}</p>\n<!-- /wp:paragraph -->\n')

    # 9. Add external links
    if article_dict.get('external_links'):
        external_links = article_dict['external_links'].strip()
        if external_links:
            if '<!-- wp:heading -->' in external_links and '<h2>Additional Resources</h2>' in external_links:
                content.append(external_links)
            else:
                if external_links.startswith('## Additional Resources'):
                    external_links = re.sub(r'^## Additional Resources\s*', '', external_links, flags=re.MULTILINE)
                    content.append('<!-- wp:heading -->\n<h2>Additional Resources</h2>\n<!-- /wp:heading -->\n')
                    links = external_links.strip().split('\n')
                    for link in links:
                        if link.strip().startswith('- ['):
                            match = re.match(r'- \[(.*?)\]\((.*?)\)(.*)', link.strip())
                            if match:
                                title = match.group(1)
                                url = match.group(2)
                                snippet = match.group(3).strip()
                                content.append('<!-- wp:paragraph -->')
                                content.append(f'<p><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a><br>{snippet}</p>')
                                content.append('<!-- /wp:paragraph -->\n')
                        elif link.strip():
                            content.append(f'<!-- wp:paragraph -->\n<p>{link.strip()}</p>\n<!-- /wp:paragraph -->\n')
                else:
                    content.append(external_links)

    # 10. Add random key takeaways if configured
    if config.keytakeaways_position == "random" and article_dict.get('block_notes'):
        content = check_and_add_key_takeaways(content, article_dict).copy()

    return ''.join(content)

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
    provider.info("Converting WordPress content to Markdown...")
    provider.debug(f"Input content length: {len(content)}")

    # Remove WordPress block comments and tags
    content = re.sub(r'<!-- wp:paragraph -->\s*<p>', '', content)
    content = re.sub(r'</p>\s*<!-- /wp:paragraph -->', '', content)
    content = re.sub(r'<!-- wp:heading -->\s*<h2>', '## ', content)
    content = re.sub(r'</h2>\s*<!-- /wp:heading -->', '', content)
    content = re.sub(r'<!-- wp:heading {"level":3} -->\s*<h3>', '### ', content)
    content = re.sub(r'</h3>\s*<!-- /wp:heading -->', '', content)

    # Convert bold/strong tags to <b> tags
    content = re.sub(r'<strong>(.*?)</strong>', r'<b>\1</b>', content)

    provider.success(f"Conversion complete (output length: {len(content)})")
    return content

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
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
        from article_generator.chunking_utils import chunk_article_for_processing, combine_chunk_results, combine_chunk_results_with_llm

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
                articleaudience=context.config.articleaudience,
                article_content=article_text,
                keyword=article_content.get('keyword', 'this topic')
            )

            # Track token usage if enabled
            if enable_token_tracking and track_token_usage:
                tokens_used = context.count_message_tokens({"role": "user", "content": prompt})
                provider.token_usage(f"Block notes generation - Request tokens: {tokens_used}")

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

                # Make the API call with all parameters
                response = make_openrouter_api_call(
                    messages=messages,
                    model=context.config.summary_keynotes_model,
                    api_key=context.config.openrouter_api_key,
                    site_url=context.config.openrouter_site_url or "https://example.com",
                    site_name=context.config.openrouter_site_name or "AI Article Generator",
                    temperature=context.config.block_notes_temperature,
                    max_tokens=max_tokens,
                    seed=getattr(context.config, 'block_notes_seed', None) if context.config.enable_seed_control else None,
                    top_p=context.config.block_notes_top_p,
                    frequency_penalty=context.config.block_notes_frequency_penalty,
                    presence_penalty=context.config.block_notes_presence_penalty
                )
                
                

                chunk_keynotes = response.choices[0].message.content.strip()

                # Add response to context
                context.add_message("assistant", chunk_keynotes)
            else:
                # Use the unified generate_completion function that supports both OpenAI and OpenRouter
                # Determine which model to use based on configuration flags
                if use_separate_model:
                    # Use the separate model for keynotes if enabled
                    model_to_use = context.config.summary_keynotes_model
                    provider.info(f"Using separate summary/keynotes model: {model_to_use}")
                else:
                    # Otherwise, determine which model to use based on whether OpenRouter is enabled
                    model_to_use = context.config.openrouter_model if (hasattr(context.config, 'use_openrouter') and context.config.use_openrouter and context.config.openrouter_api_key) else engine
                
                chunk_keynotes = generate_completion(
                    prompt=prompt,
                    model=model_to_use,
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

        # Use the LLM to combine chunks if there are multiple chunks
        if len(chunk_results) > 1:
            provider.info("Using LLM to combine keynotes chunks") 
            block_notes = combine_chunk_results_with_llm(
                results=chunk_results,
                context=context,
                combine_prompt=combine_prompt,
                is_summary=False
            )      
        else:
            block_notes = chunk_results[0]

        provider.success(f"Block notes generated successfully (length: {len(block_notes)})")
        return block_notes

    except Exception as e:
        provider.error(f"Error generating block notes: {str(e)}")
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
    provider.info("Converting content to Markdown...")
    
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
