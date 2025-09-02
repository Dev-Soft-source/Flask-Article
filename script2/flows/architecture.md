# Script2 Architecture and Flow Documentation

## System Architecture

```mermaid
graph TB
    subgraph Main Entry
        main.py --> ArticleGenerator
    end

    subgraph Core Components
        ArticleGenerator --> ContentGenerator
        ArticleGenerator --> ImageHandler
        ArticleGenerator --> PAAHandler
        ArticleGenerator --> WordPressHandler
    end

    subgraph Utils
        APIValidator[api_utils.py<br/>APIValidator]
        RetryHandler[api_utils.py<br/>RetryHandler]
        SerpAPIRotator[api_utils.py<br/>SerpAPIRotator]
        CSVProcessor[csv_utils.py<br/>CSVProcessor]
        TextProcessor[text_utils.py<br/>TextProcessor]
        DebugUtils[debug_utils.py<br/>Logging & Display]
    end

    subgraph Configuration
        Config[config.py<br/>Settings & Constants]
    end

    ContentGenerator --> ArticleContext
    ContentGenerator --> TextProcessor
    ContentGenerator --> RetryHandler
    ContentGenerator --> DebugUtils

    PAAHandler --> SerpAPIRotator
    PAAHandler --> ContentGenerator
    PAAHandler --> TextProcessor
    PAAHandler --> DebugUtils

    ImageHandler --> RetryHandler
    ImageHandler --> TextProcessor

    WordPressHandler --> RetryHandler

    ArticleGenerator --> CSVProcessor
    ArticleGenerator --> APIValidator
    ArticleGenerator --> DebugUtils
```

## Content Generation Flow

```mermaid
sequenceDiagram
    participant Main as main.py
    participant AG as ArticleGenerator
    participant CG as ContentGenerator
    participant PAA as PAAHandler
    participant WP as WordPressHandler
    participant CSV as CSVProcessor

    Main->>AG: Initialize
    Main->>AG: validate_apis()
    Main->>AG: process_csv(file_path)
    
    AG->>CSV: process_file()
    CSV-->>AG: article_data

    loop For each article
        AG->>CG: generate_title(keyword)
        AG->>CG: generate_outline(keyword)
        AG->>CG: generate_introduction(keyword, title)
        
        loop For each section
            AG->>CG: generate_paragraph(keyword, subtitle)
        end

        AG->>CG: generate_conclusion(keyword, title)

        opt if PAA enabled
            AG->>PAA: generate_paa_section(keyword)
        end

        opt if FAQ enabled
            AG->>CG: generate_faq(keyword)
        end

        opt if WordPress enabled
            AG->>WP: create_post(article)
        end
    end
```

## Article Context Management

```mermaid
graph LR
    subgraph ArticleContext
        Messages[Messages List]
        Parts[Article Parts]
        TokenCounter[Token Counter]
    end

    subgraph Article Parts
        Title[Title]
        Outline[Outline]
        Intro[Introduction]
        Sections[Sections]
        Conclusion[Conclusion]
        FAQ[FAQ]
        PAA[People Also Ask]
    end

    ContentGenerator -->|Uses| ArticleContext
    PAAHandler -->|Shares| ArticleContext
```

## File Structure

```mermaid
graph TD
    subgraph Root
        main.py
        config.py
        requirements.txt
    end

    subgraph article_generator
        content_generator.py
        image_handler.py
        paa_handler.py
        wordpress_handler.py
        article_context.py
    end

    subgraph utils
        api_utils.py
        csv_utils.py
        text_utils.py
        debug_utils.py
    end

    main.py -->|imports| article_generator
    main.py -->|imports| utils
    article_generator -->|imports| utils
```

## Configuration Management

```mermaid
graph TB
    subgraph config.py
        API_Keys[API Keys & Credentials]
        Toggles[Feature Toggles]
        Limits[Token Limits]
        Settings[Article Settings]
    end

    subgraph Feature Toggles
        add_summary
        add_faq
        add_images
        add_youtube
        add_external_links
        add_paa
        enable_markdown
        enable_wordpress
    end

    subgraph Article Settings
        language
        audience
        size_headings
        size_sections
        voice_tones
        point_of_view
    end

    API_Keys --> |Used by| Components[All Components]
    Toggles --> |Controls| Features[Feature Execution]
    Settings --> |Configures| Generation[Content Generation]
```

## Error Handling and Retry Logic

```mermaid
graph TB
    subgraph RetryHandler
        max_retries
        initial_delay
        exponential_base
        jitter
    end

    subgraph API Calls
        OpenAI[OpenAI API]
        WordPress[WordPress API]
        SerpAPI[SerpAPI]
        YouTube[YouTube API]
        Unsplash[Unsplash API]
    end

    RetryHandler -->|Manages| API_Calls[All API Calls]
    RetryHandler -->|Implements| Backoff[Exponential Backoff]
    RetryHandler -->|Handles| Errors[Error Recovery]
```

This documentation provides a comprehensive overview of how the different components in script2 interact with each other. The diagrams show:

1. Overall system architecture and component relationships
2. Sequential flow of the content generation process
3. How article context is managed and shared
4. File structure and organization
5. Configuration management and feature toggles
6. Error handling and retry mechanisms

Each component is designed to be modular and handle specific responsibilities:
- `ContentGenerator`: Core content generation logic
- `PAAHandler`: Manages "People Also Ask" content
- `ImageHandler`: Handles image processing and optimization
- `WordPressHandler`: Manages WordPress integration
- `ArticleContext`: Maintains context and manages tokens
- Utility classes: Provide supporting functionality 