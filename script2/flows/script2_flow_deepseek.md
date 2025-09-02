# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

```mermaid
flowchart TD
    A[Script2: AI-Powered Article Generator] --> B[Google Drive Mount]
    A --> C[Dependencies Installation]
    A --> D[Core Modules]
    A --> E[Workflow]
    A --> F[Output Handling]
    
    B --> B1[Authorization]
    B --> B2[File Access]
    
    C --> C1[openai]
    C --> C2[bs4]
    C --> C3[pyunsplash]
    C --> C4[google-search-results]
    C --> C5[nltk]
    C --> C6[people_also_ask]
    C --> C7[requests]
    C --> C8[random]
    
    D --> D1[Humanization Engine]
    D --> D2[GPT-3 Interface]
    D --> D3[Content Enhancer]
    D --> D4[WordPress Publisher]
    D --> D5[PAA Generator]
    
    E --> E1[CSV Input]
    E --> E2[Content Generation]
    E --> E3[Content Enhancement]
    E --> E4[Post Assembly]
    E --> E5[WordPress Integration]
```

## Core Package Architecture

```mermaid
flowchart LR
    GPTOps[GPT-3 Operations] --> |Uses| OpenAI[openai.API]
    ContentOps[Content Processing] --> |Uses| NLTK[nltk.tokenize]
    ImageOps[Image Handling] --> |Uses| Unsplash[pyunsplash]
    WebOps[Web Interactions] --> |Uses| SERP[google-search-results]
    WebOps --> |Uses| YouTube[googleapiclient.discovery]
    Formatting --> |Uses| BS4[bs4.BeautifulSoup]
```

## Main Workflow Breakdown

```mermaid
flowchart TD
    Start[CSV Input] --> Process[Parse Structured Data]
    Process --> Generate[AI Content Generation]
    Generate --> Enhance[Content Enhancement]
    Enhance --> Format[WordPress Formatting]
    Format --> Publish[REST API Posting]
    
    subgraph Generation Process
        Generate --> GPT[GPT-3 Request]
        GPT --> Humanize[Humanization Module]
        Humanize --> Grammar[Grammar Check]
    end
    
    subgraph Enhancement Process
        Enhance --> PAA[People Also Ask]
        Enhance --> IMG[Unsplash Images]
        Enhance --> YT[YouTube Embeds]
        Enhance --> FAQ[FAQ Generation]
    end
```

## Key Dependency Matrix

| Package               | Usage                                | Critical Functions                 |
|-----------------------|--------------------------------------|-------------------------------------|
| `openai`              | GPT-3 API interactions               | gpt3Request, generate_paragraph    |
| `pyunsplash`          | Unsplash image retrieval             | image_operation_unsplash           |
| `google-search-results` | SERP data collection               | serp, get_paa_questions            |
| `people_also_ask`     | Related question generation          | paa_fun, paaBreak                  |
| `nltk`                | Text processing                      | split_text_into_sentences          |
| `requests`            | WordPress REST API communication     | Post article to WP                 |
| `bs4`                 | HTML content cleaning                | Format generated content           |

## Content Generation Flow

```mermaid
flowchart LR
    CSV[CSV Input] --> Parse[Parse Keywords/Subtitles]
    Parse --> Title[Generate Title]
    Parse --> Intro[Create Introduction]
    Intro --> Sections[Generate Content Sections]
    Sections --> Humanize[Humanize Content]
    Humanize --> Enhance[Add Enhancements]
    
    Enhance --> |Conditional| PAA[PAA Section]
    Enhance --> |Conditional| IMG[Random Images]
    Enhance --> |Conditional| YT[YouTube Embed]
    Enhance --> |Conditional| FAQ[FAQ Section]
    
    Enhance --> Assemble[Final Assembly]
    Assemble --> WP[WordPress Formatting]
```

## Critical API Services

```mermaid
flowchart LR
    APIs[External Services] --> OpenAIAPI[OpenAI GPT-3]
    APIs --> GoogleAPI[Google Custom Search]
    APIs --> UnsplashAPI[Unsplash Library]
    APIs --> YouTubeAPI[YouTube Data v3]
    APIs --> WordPressAPI[WP REST API]
    
    OpenAIAPI --> |Text Generation| Core[Article Content]
    GoogleAPI --> |SERP Data| PAA[Related Questions]
    UnsplashAPI --> |Images| IMG[Article Images]
    YouTubeAPI --> |Videos| YT[Embedded Content]
    WordPressAPI --> |Publishing| WP[Live Posts]
```

## Data Flow Diagram

```mermaid
flowchart TD
    CSVFile[Structured CSV] --> Parser[CSV Parser]
    Parser --> DataDict{Data Dictionary}
    DataDict --> Keyword[Main Keyword]
    DataDict --> Subtitles[Subtitles]
    DataDict --> Images[Image Commands]
    
    Keyword --> TitleGen[Title Generation]
    Subtitles --> ContentGen[Section Generation]
    Images --> ImgProc[Image Processing]
    
    TitleGen --> Assembly[Post Assembly]
    ContentGen --> Assembly
    ImgProc --> Assembly
    
    Assembly --> WPFormat[WordPress XML-RPC Format]
    WPFormat --> APIPost[REST API Request]
```

## Error Handling Mechanism

```mermaid
flowchart TD
    Attempt[API Call] --> Success{Success?}
    Success -->|Yes| Process[Process Result]
    Success -->|No| RetryCheck{Retries Left?}
    RetryCheck -->|Yes| Wait[Exponential Backoff]
    Wait --> Retry[Retry Attempt]
    RetryCheck -->|No| Fail[Graceful Failure]
    
    subgraph Retry Logic
        Retry --> |Max 5 attempts| Success
        Wait --> |Jittered Delay| Retry
    end
```

## Content Enhancement Subsystem

```mermaid
flowchart TD
    MainContent[Generated Content] --> Enhancer[Enhancement Router]
    
    Enhancer --> |Random Images| ImgHandler[Image Handler]
    Enhancer --> |PAA Section| PAAHandler[PAA Generator]
    Enhancer --> |YouTube| YTHandler[Video Finder]
    Enhancer --> |External Links| SERPHandler[SERP Lookup]
    
    ImgHandler --> Unsplash[Unsplash API]
    PAAHandler --> GooglePAA[People Also Ask]
    YTHandler --> YouTubeAPI[Search API]
    SERPHandler --> GoogleSERP[SERP API]
    
    AllEnhancements --> Combined[Enhanced Content]
```

## Conclusion Flow

```mermaid
flowchart LR
    Final[Final Content] --> WP[WordPress Post]
    WP --> |Success| Live[Live Article]
    WP --> |Failure| Log[Error Logging]
    
    Live --> |Contains| CT[Clean Text]
    Live --> |Contains| IMG[Optimized Images]
    Live --> |Contains| MEDIA[Embedded Media]
    Live --> |Contains| SEO[SEO Elements]
```

# Key Technical Components

1. **Natural Language Processing**
   - GPT-3 API for content generation
   - NLTK for sentence tokenization
   - Custom humanization algorithms

2. **Media Handling**
   - Unsplash image integration
   - YouTube video embedding
   - Random image position logic

3. **Content Enhancement**
   - People Also Ask (PAA) section generator
   - Automated FAQ builder
   - SERP-based external links

4. **WordPress Integration**
   - REST API posting
   - Featured media handling
   - Slug generation from keywords

5. **Error Handling**
   - Exponential backoff retry logic
   - Graceful API failure handling
   - Content validation checks

# Critical Dependencies

- **AI Services**: OpenAI API, Google PAA
- **Media Services**: Unsplash API, YouTube API
- **NLP Tools**: NLTK, BeautifulSoup
- **WordPress**: XML-RPC Client, REST API
- **Utility**: Requests, Random, OS, CSV

# Potential Failure Points

1. API Rate Limiting (OpenAI/Google)
2. Media API Authentication Failures
3. CSV Formatting Errors
4. WordPress Connection Issues
5. Content Validation Failures
6. Randomization Logic Errors
