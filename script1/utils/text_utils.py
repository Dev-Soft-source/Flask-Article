# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import nltk
from typing import List
import re

def get_text_length(text: str) -> int:
    """
    Get the word count of a text.
    
    Args:
        text (str): Input text
    Returns:
        int: Number of words
    """
    words = nltk.word_tokenize(text)
    return len(words)

def clean_text(text: str) -> str:
    """
    Clean and format text by removing extra spaces, newlines, markdown formatting, etc.
    
    Args:
        text (str): Input text to clean
    Returns:
        str: Cleaned text
    """
    # Clean markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Remove bold (**text**)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Remove italic (*text*)
    text = re.sub(r'`([^`]+)`', r'\1', text)        # Remove code ticks (`text`)
    text = re.sub(r'^#+\s*', '', text)              # Remove heading hashtags
    
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    # Remove empty paragraph tags
    text = text.replace('<!-- wp:paragraph --><p></p><!-- /wp:paragraph -->', '')
    
    # Fix spacing in paragraph tags
    text = text.replace('<p> ', '<p>').replace(' </p>', '</p>')
    
    # Remove numbered list markers
    for i in range(1, 16):
        text = text.replace(f'{i}.', '')
    
    # Fix sentence spacing
    text = text.replace('.', '. ').replace('.  ', '. ')
    
    return text.strip()

def format_text_for_wordpress(text: str, tag: str = 'p') -> str:
    """
    Format text with WordPress Gutenberg block syntax and clean any markdown.
    
    Args:
        text (str): Text to format
        tag (str): HTML tag to use (default: 'p')
    Returns:
        str: Formatted text with WordPress block syntax
    """
    # Clean any markdown from the text first
    clean_content = clean_text(text)
    
    # Add newlines for better readability in the output
    if tag == 'p':
        return f'<!-- wp:{tag} -->\n<{tag}>{clean_content}</{tag}>\n<!-- /wp:{tag} -->\n'
    elif tag == 'h2':
        return f'<!-- wp:heading -->\n<h2>{clean_content}</h2>\n<!-- /wp:heading -->\n'
    elif tag == 'h3':
        return f'<!-- wp:heading {{"level":3}} -->\n<h3>{clean_content}</h3>\n<!-- /wp:heading -->\n'
    elif tag == 'h4':
        return f'<!-- wp:heading {{"level":4}} -->\n<h4>{clean_content}</h4>\n<!-- /wp:heading -->\n'
    else:
        return f'<!-- wp:{tag} -->\n<{tag}>{clean_content}</{tag}>\n<!-- /wp:{tag} -->\n'

def split_into_paragraphs(text: str, max_words_per_paragraph: int = 120) -> List[str]:
    """
    Split text into paragraphs based on word count.
    
    Args:
        text (str): Input text
        max_words_per_paragraph (int): Maximum words per paragraph
    Returns:
        List[str]: List of paragraph texts
    """
    words = text.split()
    paragraphs = []
    current_paragraph = []
    current_word_count = 0
    
    for word in words:
        if current_word_count + 1 > max_words_per_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = [word]
            current_word_count = 1
        else:
            current_paragraph.append(word)
            current_word_count += 1
    
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    
    return paragraphs

def format_paragraphs_for_wordpress(paragraphs: List[str]) -> str:
    """
    Format a list of paragraphs with WordPress block syntax.
    
    Args:
        paragraphs (List[str]): List of paragraph texts
    Returns:
        str: Formatted text with WordPress blocks
    """
    formatted_paragraphs = []
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
        formatted_paragraphs.append(format_text_for_wordpress(paragraph))
    
    return ''.join(formatted_paragraphs) 