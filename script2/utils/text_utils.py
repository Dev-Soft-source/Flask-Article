# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

import re
import html
import nltk
from typing import List, Dict
from config import Config
from nltk.tokenize import sent_tokenize, word_tokenize

class TextProcessor:
    """Handles text processing and formatting for article generation."""
    
    def __init__(self, config: Config):
        """Initialize the text processor and download required NLTK data."""
        self.config = config
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK's punkt tokenizer."""
        return sent_tokenize(text)
    
    def count_words(self, text: str) -> int:
        """Count the number of words in a text."""
        return len(word_tokenize(text))
    
    def distribute_sentences(self, sentences: List[str], min_per_paragraph: int = 2) -> List[str]:
        """Distribute sentences into paragraphs."""
        if not sentences:
            return []
            
        paragraphs = []
        current_paragraph = []
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            
            # Start a new paragraph after 2-4 sentences
            if len(current_paragraph) >= min_per_paragraph and len(current_paragraph) <= 4:
                if len(sentences) > 4 and (len(current_paragraph) >= 3 or len(paragraphs) > 0):
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                    
        # Add any remaining sentences
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
            
        return paragraphs
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    def clean_html(self, text: str) -> str:
        """Clean HTML from text and normalize it."""
        # Unescape HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Fix common formatting issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)  # Fix punctuation spacing
        text = re.sub(r'\s+([.,!?])\s*', r'\1 ', text)  # Remove space before punctuation
        text = re.sub(r'\.{3,}', '...', text)  # Normalize ellipsis
        text = re.sub(r'\s*-\s*', '-', text)  # Fix hyphen spacing
        text = re.sub(r'\(\s+', '(', text)  # Fix opening parenthesis
        text = re.sub(r'\s+\)', ')', text)  # Fix closing parenthesis
        
        # Ensure proper capitalization
        sentences = self.split_into_sentences(text)
        cleaned_sentences = [s.strip().capitalize() for s in sentences]
        
        return ' '.join(cleaned_sentences).strip()
    
    def wrap_with_paragraph_tag(self, text: str, tag: str = "p") -> str:
        """
        Wrap text with WordPress tags.
        
        Args:
            text (str): Text to wrap
            tag (str, optional): HTML tag to use. Defaults to "p".
            
        Returns:
            str: Text wrapped with WordPress tags
        """
        # Clean any markdown formatting from the text
        clean_text = text
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)  # Remove bold
        clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)      # Remove italic
        clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)        # Remove code
        
        if tag == "p":
            return f'<!-- wp:paragraph -->\n<p>{clean_text}</p>\n<!-- /wp:paragraph -->\n'
        elif tag == "h2":
            return f'<!-- wp:heading -->\n<h2>{clean_text}</h2>\n<!-- /wp:heading -->\n'
        elif tag == "h3":
            return f'<!-- wp:heading {{"level":3}} -->\n<h3>{clean_text}</h3>\n<!-- /wp:heading -->\n'
        elif tag == "h4":
            return f'<!-- wp:heading {{"level":4}} -->\n<h4>{clean_text}</h4>\n<!-- /wp:heading -->\n'
        else:
            return f'<!-- wp:paragraph -->\n<{tag}>{clean_text}</{tag}>\n<!-- /wp:paragraph -->\n'
    
    def format_heading(self, text: str, level: str = "h2") -> str:
        """
        Format text as a WordPress heading.
        
        Args:
            text (str): The heading text
            level (str): The heading level (h2, h3, etc.)
            
        Returns:
            str: Formatted WordPress heading
        """
        # Clean any markdown formatting from the heading text
        clean_text = text
        clean_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_text)  # Remove bold
        clean_text = re.sub(r'\*([^*]+)\*', r'\1', clean_text)      # Remove italic
        clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)        # Remove code
        clean_text = re.sub(r'^#+\s*', '', clean_text)              # Remove markdown headings
        
        if level == "h2":
            return f'<!-- wp:heading -->\n<h2>{clean_text}</h2>\n<!-- /wp:heading -->\n'
        elif level == "h3":
            return f'<!-- wp:heading {{"level":3}} -->\n<h3>{clean_text}</h3>\n<!-- /wp:heading -->\n'
        elif level == "h4":
            return f'<!-- wp:heading {{"level":4}} -->\n<h4>{clean_text}</h4>\n<!-- /wp:heading -->\n'
        else:
            # Default to h2 if invalid level is provided
            return f'<!-- wp:heading -->\n<h2>{clean_text}</h2>\n<!-- /wp:heading -->\n'
    
    def format_blockquote(self, text: str) -> str:
        """Format text as a WordPress blockquote."""
        return f'<!-- wp:quote --><blockquote class="wp-block-quote"><p>{text}</p></blockquote><!-- /wp:quote -->'
    
    def format_youtube_embed(self, video_id: str) -> str:
        """Format YouTube video embed code."""
        return f'''<!-- wp:embed {{"url":"https://www.youtube.com/watch?v={video_id}","type":"video","providerNameSlug":"youtube","responsive":true,"className":"wp-embed-aspect-16-9 wp-has-aspect-ratio"}} -->
<figure class="wp-block-embed is-type-video is-provider-youtube wp-block-embed-youtube wp-embed-aspect-16-9 wp-has-aspect-ratio">
<div class="wp-block-embed__wrapper">https://www.youtube.com/watch?v={video_id}</div>
</figure>
<!-- /wp:embed -->'''
    
    def format_image(self, url: str, alt: str = "") -> str:
        """Format image as WordPress media block."""
        return f'''<!-- wp:image {{"align":"center","sizeSlug":"large"}} -->
<figure class="wp-block-image aligncenter size-large">
<img src="{url}" alt="{alt}"/>
</figure>
<!-- /wp:image -->''' 