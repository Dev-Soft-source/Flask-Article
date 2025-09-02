# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

# AI Article Generator (Script2)

An advanced AI-powered article generation system that creates high-quality, SEO-optimized content from structured CSV input. The system uses OpenAI's GPT models for content generation and supports various enhancement features including FAQ generation, "People Also Ask" sections, image handling, and WordPress integration.

## Features

- 📝 AI-powered article generation using OpenAI GPT
- 📊 CSV-based structured content input
- ❓ Automatic FAQ generation
- 🔍 "People Also Ask" section integration
- 🖼️ Image handling with Unsplash
- 🎥 YouTube video embedding support
- 🔗 External reference links
- 📱 WordPress publishing
- 🎨 Markdown output support
- 🛠️ Configurable content enhancement options

## Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) Unsplash API key for images
- (Optional) SerpAPI key for PAA content
- (Optional) WordPress site with API access

## Virtual Environment Setup

It's recommended to run this project in a virtual environment. Here's how to set it up on different operating systems:

### Windows

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# If using PowerShell and getting execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### macOS/Linux

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Verify Activation
After activation, you should see `(venv)` in your terminal prompt. You can verify Python location:
```bash
which python  # macOS/Linux
where python  # Windows
```

### Deactivate
When you're done, you can deactivate the virtual environment:
```bash
deactivate
```

## Installation

1. Clone the repository
2. Create and activate virtual environment (see above)
3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure your API keys in `config.py`:
```python
# API Keys and Credentials
OPENAI_API_KEY = "your_openai_key"
OPENAI_MODEL = "gpt-4-turbo-preview"  # or your preferred model
IMAGE_API_KEY = "your_image_key"  # Optional, for images
YOUTUBE_API_KEY = "your_youtube_key"    # Optional, for videos
SERP_API_KEYS = [                       # Optional, for PAA content
    "your_serp_api_key1",
    "your_serp_api_key2"
]

# WordPress Settings (Optional)
WP_URL = "your_wp_url"
WP_USER = "your_wp_user"
WP_APP_PASS = "your_wp_app_password"
WP_CATEGORIES = "1"
WP_AUTHOR = "2"
WP_POST_STATUS = "draft"
```

## Usage

1. Prepare your input CSV file with the following structure:
```csv
keyword,featured_img,subtitle1,img1,subtitle2,img2...
can domestic cats live outside,outdoor cat,Natural Adaptations of Cats for Outdoor Living,cat hunting,Health Risks and Challenges,sick cat...
```

2. Run the script:
```bash
python main.py --input your_input.csv
```

## Configuration

Edit `config.py` to customize:

### Feature Toggles
```python
FEATURE_TOGGLES = {
    'add_summary': True,        # Add summary to the article
    'add_faq': True,           # Add FAQ section
    'add_images': False,       # Add images from Unsplash
    'add_youtube_video': False, # Add YouTube videos
    'add_external_links': False, # Add reference links
    'add_paa_paragraphs': True, # Add "People Also Ask" sections
    'add_block_notes': False,   # Add block notes
    'correct_grammar': False,   # Grammar correction
    'humanize_text': False,     # Text humanization
    'randomize_images': True,   # Random image selection
    'enable_markdown_save': True, # Save as markdown
    'enable_wordpress_publish': False # WordPress publishing
}
```

### Article Settings
```python
ARTICLE_CONFIG = {
    'language': 'English',
    'audience': 'general wide used',
    'size_headings': 3,    # Number of headings (1-15)
    'size_sections': 3,    # Sections per paragraph (1-5)
}
```

## Project Structure

```
script2/
├── article_generator/        # Core generation components
│   ├── content_generator.py  # Main content generation
│   ├── image_handler.py     # Image processing
│   ├── paa_handler.py       # PAA content
│   ├── wordpress_handler.py # WordPress integration
│   └── article_context.py   # Context management
├── utils/                   # Utility modules
│   ├── api_utils.py        # API handling
│   ├── csv_utils.py        # CSV processing
│   ├── text_utils.py       # Text processing
│   └── debug_utils.py      # Logging & display
├── flows/                   # Documentation
├── generated_articles/      # Output directory
├── img/                     # Image storage
├── config.py               # Configuration
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

## Output

The script generates:
1. Markdown files in `generated_articles/` directory
2. (Optional) WordPress posts if publishing is enabled
3. Console output showing generation progress and content

## Error Handling

- Automatic retry with exponential backoff for API calls
- Comprehensive error logging
- Token usage tracking
- Validation for all API keys and connections

## Architecture Documentation

See `flows/architecture.md` for detailed system architecture and flow diagrams.

## Contributing

Feel free to submit issues and enhancement requests!
