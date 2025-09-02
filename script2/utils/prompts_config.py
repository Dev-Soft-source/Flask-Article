"""Configuration class for article generation prompts."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Prompts:
    """Dataclass for managing prompt templates."""
    title: str
    title_craft: str
    outline: str
    introduction: str
    paragraph: str
    paragraph_with_heading: str
    conclusion: str
    faq: str
    system_message: str
    meta_description: str
    wordpress_excerpt: str
    grammar: str 
    humanize: str 
    blocknote: str 
    summarize: str
    paa_answer: str
    summary_combine: str
    blocknotes_combine: str
    
    def format_prompt(self, prompt_type: str, **kwargs: Any) -> str:
        """Format a prompt template with provided variables."""
        prompt_map = {
            'title': self.title,
            'title_craft': self.title_craft,
            'outline': self.outline,
            'introduction': self.introduction,
            'paragraph': self.paragraph,
            'paragraph_with_heading': self.paragraph_with_heading,
            'conclusion': self.conclusion,
            'faq': self.faq,
            'system_message': self.system_message,
            'meta_description': self.meta_description,
            'wordpress_excerpt': self.wordpress_excerpt,
            'grammar': self.grammar,
            'humanize': self.humanize,
            'blocknote': self.blocknote,
            'summarize': self.summarize,
            'paa_answer': self.paa_answer,
            'summary_combine': self.summary_combine,
            'blocknotes_combine': self.blocknotes_combine,
        }
        
        if prompt_type not in prompt_map:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        return prompt_map[prompt_type].format(**kwargs)
    
    def __post_init__(self):
        """Validate that all required prompts are present."""
        required_prompts = [
            'title',
            'title_craft',
            'outline',
            'introduction',
            'paragraph',
            'paragraph_with_heading',
            'conclusion',
            'faq',
            'system_message',
            'meta_description',
            'wordpress_excerpt',
            'grammar',
            'humanize',
            'blocknote',
            'summarize',
            'paa_answer',
            'summary_combine',
            'blocknotes_combine'
        ]
        
        for prompt in required_prompts:
            if not getattr(self, prompt):
                raise ValueError(f"Missing required prompt: {prompt}")
