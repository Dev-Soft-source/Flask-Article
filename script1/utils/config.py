# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define constant dictionaries for configuration options
VOICE_TONES = {
    "professional": "Authoritative, precise language appropriate for business or academic contexts",
    "formal": "Proper, structured writing with minimal contractions or colloquialisms",
    "casual": "Relaxed, conversational tone with simple language",
    "friendly": "Warm, approachable style that connects personally with the reader",
    "persuasive": "Compelling language focused on convincing the reader"
}

ARTICLE_TYPES = {
    "Default": "Standard informational article",
    "How-To Guide": "Step-by-step instructions to accomplish a task",
    "Listicle": "List-based article with numbered points",
    "Product Review": "Evaluation of products or services",
    "News": "Timely reporting on recent events or developments",
    "Case Study": "In-depth analysis of a specific example or scenario",
    "Opinion": "Subjective viewpoint or argument on a topic",
    "Tutorial": "Educational content teaching specific skills",
    "Question And Answer": "Format addressing common questions directly"
}

ARTICLE_AUDIENCES = {
    "General": "Accessible to most readers with minimal specialized knowledge",
    "Beginners": "Assumes no prior knowledge, explains basic concepts",
    "Intermediate": "Some familiarity with the topic, moderate technical language",
    "Advanced": "Strong background knowledge, can handle complex concepts",
    "Expert": "Deep subject expertise, highly technical and specialized content"
}

POINT_OF_VIEWS = {
    "First Person - Singular": "Personal perspective using 'I' and 'me'",
    "First Person - Plural": "Collective perspective using 'we' and 'us'",
    "Second Person": "Directly addresses the reader as 'you'",
    "Third Person": "Objective perspective using 'he,' 'she,' 'they,' or 'it'"
}

@dataclass
class Config:
    # Article Generation Settings
    add_summary_into_article: bool = True
    add_faq_into_article: bool = True
    add_image_into_article: bool = True
    add_external_links_into_article: bool = True
    add_PAA_paragraphs_into_article: bool = True
    add_blocknote_into_article: bool = True
    enable_title_crafting: bool = False  # Controls whether titles should be optimized by LLM
    
    # PAA Settings
    paa_max_questions: int = 5  # Maximum number of PAA questions to display
    paa_min_questions: int = 3  # Minimum number of PAA questions when using random range
    paa_use_random_range: bool = False  # Whether to use a random range for PAA questions

    rag_openrouter_model: str = ""

    # RAG Configuration
    rag_article_retriever_engine: str= "Duckduckgo"
    enable_rag: bool = True
    rag_chunk_size: int = 500
    rag_num_chunks: int = 3
    rag_cache_dir: str = "cache/rag_cache"
    rag_embedding_model: str = "all-MiniLM-L6-v2"
    rag_embedding_dimension: int = 384
    rag_fallback_urls: List[str] = field(default_factory=lambda: [
        "https://www.britannica.com/search?query=",
        "https://www.nationalgeographic.com/search?q=",
        "https://www.sciencedirect.com/search?qs=",
        "https://www.ncbi.nlm.nih.gov/search/all/?term=",
        "https://scholar.google.com/scholar?q="
    ])
    

    # Meta Description Settings
    enable_meta_description: bool = True
    meta_description_max_length: int = 155
    meta_description_min_length: int = 120
    meta_description_temperature: float = 0.7
    meta_description_top_p: float = 1.0
    meta_description_frequency_penalty: float = 0.0
    meta_description_presence_penalty: float = 0.0
    enable_rag_search_engine: bool = False

    # Temperature and Generation Control Settings
    content_generation_temperature: float = 1.0
    content_generation_top_p: float = 1.0
    content_generation_frequency_penalty: float = 0.0
    content_generation_presence_penalty: float = 0.0
    content_generation_max_tokens: int = 700  # Default max tokens for content generation

    humanization_temperature: float = 0.7
    humanization_top_p: float = 1.0
    humanization_frequency_penalty: float = 0.0
    humanization_presence_penalty: float = 0.0
    # Note: The system uses a dynamic token calculation approach where max_tokens is set to
    # 120% of the input text length rather than a fixed value. This provides more efficient
    # token usage while ensuring sufficient room for text expansion.
    humanization_max_tokens: int = 700  # Default max tokens for humanization

    grammar_check_temperature: float = 0.3
    grammar_check_top_p: float = 1.0
    grammar_check_frequency_penalty: float = 0.0
    grammar_check_presence_penalty: float = 0.0
    # Note: Grammar checking also uses dynamic token calculation (120% of input text length)
    # rather than a fixed maximum. This ensures efficient token usage while providing enough
    # room for corrections.
    grammar_check_max_tokens: int = 700  # Default max tokens for grammar check

    block_notes_temperature: float = 0.7
    block_notes_top_p: float = 1.0
    block_notes_frequency_penalty: float = 0.0
    block_notes_presence_penalty: float = 0.0
    block_notes_max_tokens: int = 300  # Default max tokens for block notes

    faq_generation_temperature: float = 0.7
    faq_generation_top_p: float = 1.0
    faq_generation_frequency_penalty: float = 0.0
    faq_generation_presence_penalty: float = 0.0
    faq_generation_max_tokens: int = 1000  # Default max tokens for FAQ generation

    # Content-specific max tokens settings
    title_max_tokens: int = 100  # Maximum tokens for title generation
    outline_max_tokens: int = 500  # Maximum tokens for outline generation
    introduction_max_tokens: int = 700  # Maximum tokens for introduction generation
    paragraph_max_tokens: int = 700  # Maximum tokens for paragraph generation
    conclusion_max_tokens: int = 700  # Maximum tokens for conclusion generation
    meta_description_max_tokens: int = 200  # Maximum tokens for meta description generation

    # Seed Control Settings
    enable_seed_control: bool = False  # Master toggle for seed control
    title_seed: Optional[int] = None
    outline_seed: Optional[int] = None  # Added outline seed
    introduction_seed: Optional[int] = None
    paragraph_seed: Optional[int] = None
    conclusion_seed: Optional[int] = None
    key_takeaways_seed: Optional[int] = None
    faq_seed: Optional[int] = None
    meta_description_seed: Optional[int] = None

    # Feature Toggles
    enable_grammar_check: bool = False
    enable_text_humanization: bool = False
    enable_progress_display: bool = True
    enable_token_tracking: bool = True
    enable_image_generation: bool = True
    enable_youtube_videos: bool = False
    enable_wordpress_upload: bool = False
    enable_markdown_save: bool = True
    enable_context_save: bool = False  # New option to save ArticleContext to a file

    # YouTube Settings
    youtube_video_width: int = 560
    youtube_video_height: int = 315
    youtube_max_results: int = 1
    # youtube_position: str = "after_introduction"  # after_introduction, after_first_section, end
    
    # positioning
    youtube_position:str = "random", #random / first_heading /end
    keytakeaways_position:str = "random", #random / before_conclusion / middle
    image_position:str = "end" #random / under_first heading / middle / end
    paa_image_position:str = "first_heading" #first_heading /end

    # Image Settings
    max_number_of_images: int = 20
    orientation: str = "landscape"
    order_by: str = "relevant"
    # Image alignment options: "aligncenter", "alignleft", "alignright"
    image_alignment: str = "aligncenter"
    # Image compression options
    enable_image_compression: bool = False
    image_compression_quality: int = 70  # 0-100, higher is better quality but larger file size
    # Prevent duplicate images in the same article
    prevent_duplicate_images: bool = True

    # Voice Tone Settings
    voicetone: str = "professional"  # Default tone
    available_voice_tones: Dict[str, str] = field(default_factory=lambda: VOICE_TONES)

    # Article Structure Settings
    sizesections: int = 3 # MAIN SECTIONS
    sizeheadings: int = 4 # SUBSECTIONS

    # Article Size Settings
    section_token_limit: int = 2000
    paragraphs_per_section: int = 2
    min_paragraph_tokens: int = 300
    max_paragraph_tokens: int = 500
    
    # Paragraph Heading Settings
    enable_paragraph_headings: bool = True
    max_paragraph_headings_per_section: int = 5  # Maximum number of paragraph headings per section
    refine_paragraph_headings: bool = True  # Whether to allow the LLM to refine outline-based headings
    variable_paragraph_headings: bool = False  # Whether to use a variable number of headings
    
    # Variable Paragraphs Settings
    enable_variable_paragraphs: bool = False  # Enable random number of paragraphs for natural structure
    min_variable_paragraphs: int = 1  # Minimum paragraphs when using variable mode
    max_variable_paragraphs: int = 5  # Maximum paragraphs when using variable mode

    # Article Type Settings
    articletype: str = "Default"  # Default article type
    available_article_types: Dict[str, str] = field(default_factory=lambda: ARTICLE_TYPES)

    # Article Audience Settings
    articleaudience: str = "General"  # Default audience
    available_article_audiences: Dict[str, str] = field(default_factory=lambda: ARTICLE_AUDIENCES)

    # Language and Point of View Settings
    articlelanguage: str = "English"
    pointofview: str = "Third Person"  # Default point of view
    available_point_of_views: Dict[str, str] = field(default_factory=lambda: POINT_OF_VIEWS)

    # Bold and Italics, etc into Articles Settings
    add_bold_into_article: bool = True
    add_lists_into_articles: bool = True
    add_tables_into_articles: bool = True
    enable_variable_subheadings: bool = True
    add_italic_into_article: bool = True
    
    # API Keys and Credentials
    openai_key: str = field(default_factory=lambda: os.getenv('OPENAI_API_KEY', ''))
    openai_org_id: str = field(default_factory=lambda: os.getenv('OPENAI_ORG_ID', ''))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv('OPENROUTER_API_KEY', ''))
    use_openrouter: bool = field(default_factory=lambda: os.getenv('USE_OPENROUTER', 'false').lower() == 'true')
    openrouter_site_url: str = field(default_factory=lambda: os.getenv('OPENROUTER_SITE_URL', ''))
    openrouter_site_name: str = field(default_factory=lambda: os.getenv('OPENROUTER_SITE_NAME', ''))
    youtube_api: str = field(default_factory=lambda: os.getenv('YOUTUBE_API_KEY', ''))
    
    # Image settings 
    image_source: str = "Stock"
    stock_primary_source: str = "pixabay"
    secondary_source_image: bool = True
    image_api:bool = True
    huggingface_model:str = "stabilityai/stable-diffusion-xl-base-1.0"
    
    # Image captioning settings 
    image_caption_instance:str = "openai/clip-vit-base-patch32"

    
    # Image sources api keysB
    unsplash_api_key:str = field(default_factory=lambda: os.getenv('UNSPLASH_API_KEY', ''))
    pexels_api_key:str = field(default_factory=lambda: os.getenv('PEXELS_API_KEY', ''))
    pixabay_api_key:str = field(default_factory=lambda: os.getenv('PIXABAY_API_KEY', ''))
    giphy_api_key:str = field(default_factory=lambda: os.getenv('GIPHY_API_KEY', ''))
    huggingface_api_key:str = field(default_factory=lambda: os.getenv('HUGGINGFACE_API_KEY', ''))
    
    
   
    openai_model: str = "gpt-4o-mini-2024-07-18"
    openai_token: int = 4096
    max_context_window_tokens: int = 128000
    token_padding: int = 100
    warn_token_threshold: float = 0.8

    # WordPress Settings
    # Sensitive information from .env
    website_name: str = field(default_factory=lambda: os.getenv('WP_WEBSITE_NAME', ''))
    Username: str = field(default_factory=lambda: os.getenv('WP_USERNAME', ''))
    App_pass: str = field(default_factory=lambda: os.getenv('WP_APP_PASS', ''))
    # Configuration elements moved from .env to config class
    categories: str = "1"  # Default category ID
    author: str = "1"  # Default author ID
    custom_author: str = ""  # Optional custom author ID
    status: str = "draft"  # Default post status (draft or publish)

    # API Settings
    serp_api_key: str = field(default_factory=lambda: os.getenv('SERPER_API_KEY', '') or os.getenv('SERP_API_KEY', ''))

    # Subtitle Formatting
    subtitle_levels: Dict[str, str] = None

    # API Retry Settings
    initial_delay: int = 1
    exponential_base: int = 2
    jitter: bool = True
    max_retries: int = 10

    # Markdown Settings
    markdown_output_dir: str = "generated_articles"

    # Rate Limiting Settings
    enable_rate_limiting: bool = True
    openai_rpm: int = 60  # Requests per minute for OpenAI
    openai_rpd: int = 10000  # Requests per day for OpenAI
    serpapi_rpm: int = 5  # Requests per minute for SerpAPI
    serpapi_rpd: int = 100  # Requests per day for SerpAPI
    unsplash_rpm: int = 50  # Requests per minute for Unsplash
    unsplash_rpd: int = 5000  # Requests per day for Unsplash
    youtube_rpm: int = 100  # Requests per minute for YouTube
    youtube_rpd: int = 10000  # Requests per day for YouTube
    rate_limit_cooldown: int = 60  # Cooldown period in seconds after hitting rate limit
    
    # Openverse settings (as of 2024)
    openverse_rpm: int = 60   # Openverse unofficially allows ~60 requests per minute
    openverse_rpd: int = 5000 # Conservative daily limit; official cap isn't enforced publicly

    # Pexels settings (as of 2024)
    pexels_rpm: int = 200     # Official: 200 requests per minute
    pexels_rpd: int = 20000  # Official: 20,000 requests per month for free tier → ~666/day

    # Pixabay settings (as of 2024)
    pixabay_rpm: int = 60     # Unofficial safe limit
    pixabay_rpd: int = 5000  # Default daily quota for free-tier API keys

    # DuckDuckGo settings (as of 2024)
    duckduckgo_rpm: int = 30      
    duckduckgo_rpd: int = 10000
    
    # Huggingface settings (as of 2024)
    huggingface_rpm: int = 30      
    huggingface_rpd: int = 10000

    # Context Save Settings
    context_save_dir: str = "article_contexts"  # Directory to save context files
    
    # OpenRouter Configuration
    openrouter_models: dict = None  # Dictionary of available OpenRouter models
    openrouter_model: str = "anthropic/claude-3-opus-20240229"  # Default OpenRouter model

    # Summary and Keynotes Model Settings
    enable_separate_summary_model: bool = False  # Enable using a separate model for summary and keynotes
    summary_keynotes_model: str = "anthropic/claude-3-opus-20240229"  # Default model with large context window
    summary_max_tokens: int = 800  # Maximum tokens for summary generation
    keynotes_max_tokens: int = 300  # Maximum tokens for keynotes generation
    summary_chunk_size: int = 8000  # Chunk size for summary generation when chunking is needed
    keynotes_chunk_size: int = 8000  # Chunk size for keynotes generation when chunking is needed
    summary_combination_temperature: float = 0.3  # Temperature for combining summary chunks
    summary_combination_paragraphs: int = 2  # Number of paragraphs to generate when combining summary chunks
    keynotes_combination_temperature: float = 0.3  # Temperature for combining keynotes chunks
    keynotes_combination_paragraphs: int = 1  # Number of paragraphs to generate when combining keynotes chunks
    
    # Grammar Check Model Settings
    enable_separate_grammar_model: bool = False  # Enable using a separate model for grammar checking
    grammar_check_model: str = "anthropic/claude-3-haiku-20240307"  # Default model for grammar checking
    
    # Humanization Model Settings
    enable_separate_humanization_model: bool = False  # Enable using a separate model for text humanization
    humanization_model: str = "anthropic/claude-3-sonnet-20240229"  # Default model for text humanization

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.enable_youtube_videos and not self.youtube_api:
            raise ValueError("YouTube API key is required when enable_youtube_videos is True")

        if self.enable_image_generation and not self.image_api:
            raise ValueError("Image API key is required when enable_image_generation is True")

        if self.enable_wordpress_upload and not all([self.website_name, self.Username, self.App_pass]):
            raise ValueError("WordPress credentials are required when enable_wordpress_upload is True")
        
        if self.enable_rag_search_engine == "Google":
            if self.add_PAA_paragraphs_into_article and not self.serp_api_key:
                raise ValueError("SerpAPI key is required when add_PAA_paragraphs_into_article is True")

        # Validate generation control parameters
        for param_type in ['content_generation', 'humanization', 'grammar_check', 'block_notes', 'faq_generation', 'meta_description']:
            # Temperature validation (0 to 2)
            temp = getattr(self, f"{param_type}_temperature")
            if not 0 <= temp <= 2:
                raise ValueError(f"{param_type}_temperature must be between 0 and 2")

            # Top_p validation (0 to 1)
            top_p = getattr(self, f"{param_type}_top_p")
            if not 0 <= top_p <= 1:
                raise ValueError(f"{param_type}_top_p must be between 0 and 1")

            # Frequency penalty validation (-2 to 2)
            freq_penalty = getattr(self, f"{param_type}_frequency_penalty")
            if not -2 <= freq_penalty <= 2:
                raise ValueError(f"{param_type}_frequency_penalty must be between -2 and 2")

            # Presence penalty validation (-2 to 2)
            pres_penalty = getattr(self, f"{param_type}_presence_penalty")
            if not -2 <= pres_penalty <= 2:
                raise ValueError(f"{param_type}_presence_penalty must be between -2 and 2")

        # Initialize subtitle levels if None
        if self.subtitle_levels is None:
            self.subtitle_levels = {f"Subtitle{i}": "h3" for i in range(1, 13)}
            self.subtitle_levels["Subtitle1"] = "h2"

        # Initialize OpenRouter models if None
        if self.openrouter_models is None:
            self.openrouter_models = {
                "claude-opus": "anthropic/claude-3-opus-20240229",
                "claude-sonnet": "anthropic/claude-3-sonnet-20240229",
                "claude-haiku": "anthropic/claude-3-haiku-20240307",
                "deepseek-coder": "deepseek/deepseek-coder",
                "deepseek-chat": "deepseek/deepseek-chat",
                "llama3-70b": "meta-llama/llama-3-70b-instruct",
                "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct-v0.1",
                "zephyr-chat": "huggingface/zephyr-7b-beta"
            }
