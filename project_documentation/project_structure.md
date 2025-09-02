# Project Structure Documentation

## Overview
This document provides a comprehensive overview of the AI Article Generation System, designed for handover to new developers. The system consists of two main scripts that generate SEO-optimized articles using AI, with different approaches to content structuring.

## Project Architecture

The project is organized into two main directories (`script1/` and `script2/`), each representing a different approach to article generation. Both scripts share similar core components but differ in their input methods and content structuring strategies.

## Directory Structure

```
scripts_p3/
├── script1/                          # Auto-generates article structure
│   ├── main.py                      # Entry point - initializes configuration
│   ├── article_generator/           # Core generation modules
│   │   ├── content_generator.py     # Creates individual content sections
│   │   ├── generator.py             # Orchestrates complete article generation
│   │   ├── image_handler.py         # Image sourcing and processing
│   │   ├── wordpress_handler.py     # WordPress integration
│   │   ├── paa_handler.py          # People Also Ask content
│   │   ├── rag_retriever.py        # Web content retrieval
│   │   └── ... (additional modules)
│   ├── utils/                       # Shared utilities
│   │   ├── config.py               # Configuration management
│   │   ├── prompts_config.py       # Prompt templates
│   │   └── ... (additional utilities)
│   └── keywords.txt                 # Keywords for generation
│
├── script2/                         # Manual article structure input
│   ├── main.py                      # Entry point - initializes configuration
│   ├── config.py                    # Configuration settings
│   ├── article_generator/           # Core generation modules
│   │   ├── content_generator.py     # Creates content for predefined sections
│   │   ├── generator.py             # Orchestrates article generation
│   │   └── ... (similar modules as script1)
│   ├── utils/                       # Shared utilities
│   └── input.csv                   # Keywords + predefined structure
│
├── project_documentation/           # Project documentation
├── docs/                           # Additional documentation
└── tests/                          # Test suites
```

## Core Components

### 1. Content Generation Pipeline

#### Content Generator ([`content_generator.py`](script1/article_generator/content_generator.py))
- **Purpose**: Handles AI-powered content generation for individual article sections
- **Key Functions**:
  - `generate_title()`: Creates SEO-optimized article titles
  - `generate_outline()`: Creates article structure (script1 only)
  - `generate_introduction()`: Writes article introductions
  - `generate_paragraph()`: Creates content for specific sections
  - `generate_conclusion()`: Writes article conclusions
  - `generate_faq()`: Creates FAQ sections
  - `generate_article_summary()`: Generates article summaries

#### Generator ([`generator.py`](script1/article_generator/generator.py))
- **Purpose**: Orchestrates the complete article generation process
- **Responsibilities**:
  - Coordinates between different handlers
  - Manages content flow and assembly
  - Handles WordPress integration
  - Manages image insertion and positioning

### 2. Main Entry Points

#### Script 1: Auto-Generation ([`main.py`](script1/main.py))
- **Approach**: Fully automated article generation
- **Input**: CSV file with keywords only
- **Process**:
  1. Reads keywords from CSV
  2. Auto-generates complete article structure
  3. Creates title, outline, sections, and conclusion
  4. Adds images, FAQs, and additional content
  5. Posts to WordPress

#### Script 2: Manual Structure ([`main.py`](script2/main.py))
- **Approach**: Manual article structure with automated content
- **Input**: CSV file with keywords AND predefined section structure
- **Process**:
  1. Reads keywords and section structure from CSV
  2. Uses predefined section titles instead of auto-generating
  3. Generates content for each predefined section
  4. Maintains same additional features as script1

### 3. Key Differences Between Scripts

| Aspect | Script 1 | Script 2 |
|--------|----------|----------|
| **Input Structure** | Keywords only | Keywords + section titles |
| **Outline Generation** | AI-generated | Predefined in CSV |
| **Content Control** | Fully automated | Semi-automated with manual structure |
| **CSV Format** | `keyword,image_keyword` | `keyword,featured_img,Subtitle1,img1,Subtitle2,img2,...` |
| **Use Case** | Bulk content generation | Controlled content structure |

### 4. CSV Input Formats

#### Script 1 Input Format (keywords.txt)
```
how to grow tomatoes
best indoor plants
organic gardening tips
```

#### Script 2 Input Format (input.csv)
```csv
keyword,featured_img,Subtitle1,img1,Subtitle2,img2,Subtitle3,img3,...
"growing tomatoes","tomato plant","Soil Preparation","soil prep","Watering Schedule","watering tips","Pest Control","pest management"
```

### 5. Configuration System

Both scripts use comprehensive configuration systems:

- **API Keys**: OpenAI, OpenRouter, SerpAPI, YouTube, WordPress
- **Content Settings**: Voice tone, article type, audience, point of view
- **Feature Toggles**: Images, FAQs, summaries, external links, etc.
- **Rate Limiting**: API call limits and cooldowns
- **Model Selection**: Different models for different content types

### 6. Content Features

- **Images**: Stock photos, AI-generated images, YouTube thumbnails
- **SEO Optimization**: Meta descriptions, structured content
- **Rich Content**: FAQs, key takeaways, external links
- **WordPress Integration**: Direct posting with proper formatting
- **RAG System**: Web content retrieval for enhanced context
- **People Also Ask**: Integration with Google PAA sections

## Development Notes

### Current Implementation Status
- ✅ Script 1: Fully functional with all features
- ✅ Script 2: Functional but missing subsection implementation
- ✅ Core content generation works in both scripts
- ✅ WordPress integration operational
- ✅ Image handling implemented

### Areas Needing Attention
1. **Subsection Implementation in Script 2**: Currently only handles main sections, needs subsection support like Script 1
2. **Feature Parity**: Ensure Script 2 has all features available in Script 1
3. **Error Handling**: Some edge cases need better handling
4. **Performance Optimization**: Large articles may hit token limits

### Next Steps for New Developers
1. Review the configuration files in both scripts
2. Test with sample CSV inputs
3. Implement subsection support in Script 2 (see separate documentation)
4. Review refactoring strategy documentation for code improvements
5. Check current problems documentation for known issues

## Getting Started

1. **Setup Environment**:
   ```bash
   cd script1  # or script2
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Test Run**:
   ```bash
   # Script 1
   python main.py keywords.txt
   
   # Script 2
   python main.py --input input.csv
   ```

3. **Configuration**:
   - Review `config.py` or configuration initialization in `main.py`
   - Adjust settings for your specific needs
   - Test with small batches first

## Support Files

- **Documentation**: Check `/docs/` directory for detailed feature documentation
- **Tests**: Run test suites in `/tests/` directories
- **Logs**: Monitor `/logs/` directories for debugging
- **Examples**: Review sample CSV files and configuration examples