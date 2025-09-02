# بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ

# Current Article Generation Flow (2024 Configuration)

```mermaid
graph TD
    Start([Start]) --> ReadConfig[Read Configuration<br/>Feature Toggles & Settings]
    ReadConfig --> ReadKeywords[Read keywords.txt]
    ReadKeywords --> ValidateFormat[Validate keyword format<br/>keyword,image_keyword]
    
    subgraph API_Validation
        ValidateFormat --> ValidateAPIs[Validate APIs]
        ValidateAPIs --> CheckOpenAI[Check OpenAI API<br/>GPT-4o mini model]
        ValidateAPIs --> CheckYouTube[Check YouTube API]
        ValidateAPIs --> CheckSerpAPI[Check SerpAPI Keys]
        ValidateAPIs --> CheckUnsplash[Check Unsplash API]
    end
    
    subgraph Article_Generation
        ValidateAPIs --> ProcessKeyword[Process Keyword:<br/>'can domestic cats live in the wild']
        
        ProcessKeyword --> GenerateTitle[1. Generate Title<br/>Show progress]
        GenerateTitle --> GenerateOutline[2. Generate Outline<br/>Show progress]
        GenerateOutline --> GenerateIntro[3. Generate Introduction<br/>Show progress]
        
        GenerateIntro --> GenerateSections[4. Generate Sections<br/>Show progress for each]
        GenerateSections --> GenerateConclusion[5. Generate Conclusion<br/>Show progress]
        
        GenerateConclusion --> FeatureChecks{Check Feature Toggles}
    end
    
    subgraph Feature_Processing
        FeatureChecks --> |enable_grammar_check| Grammar[Grammar Check]
        FeatureChecks --> |enable_text_humanization| Humanize[Text Humanization]
        FeatureChecks --> |enable_image_generation| Images[Get Images<br/>4-7 Unsplash images]
        FeatureChecks --> |add_youtube_video| YouTube[Add YouTube Videos]
        FeatureChecks --> |add_PAA_paraphraps| PAA[Generate PAA Section]
        FeatureChecks --> |add_external_links| Links[Add External Links]
        
        Grammar & Humanize & Images & YouTube & PAA & Links --> FormatWP[Format for WordPress]
    end
    
    subgraph WordPress_Integration
        FormatWP --> UploadMedia[Upload Media Files]
        UploadMedia --> CreatePost[Create WordPress Post<br/>Status: Draft<br/>Category: 1<br/>Author: 2]
    end
    
    CreatePost --> Success{Success?}
    Success --> |Yes| NextKeyword{More Keywords?}
    Success --> |No| Error[Log Error]
    
    NextKeyword --> |Yes| ProcessKeyword
    NextKeyword --> |No| End([End])
    Error --> End

    style Start fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style End fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style API_Validation fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Article_Generation fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Feature_Processing fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style WordPress_Integration fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
```

# Configuration Overview

```mermaid
graph TD
    subgraph Feature_Toggles
        A[Core Features] --> B[Summary: Enabled]
        A --> C[FAQ/PAA: Enabled]
        A --> D[External Links: Enabled]
        A --> E[Block Notes: Enabled]
        
        F[Media Features] --> G[Images: Enabled]
        F --> H[YouTube Videos: Enabled]
        
        I[Processing Features] --> J[Grammar Check: Disabled]
        I --> K[Text Humanization: Disabled]
        I --> L[Progress Display: Enabled]
        I --> M[Token Tracking: Enabled]
    end
    
    subgraph Content_Settings
        N[Article Structure] --> O[Single Section per Paragraph]
        P[Language] --> Q[English]
        R[Point of View] --> S[Auto Select]
    end
    
    subgraph Media_Settings
        T[Image Configuration] --> U[Min: 4 Images]
        T --> V[Max: 7 Images]
        T --> W[Landscape Orientation]
        T --> X[Random Selection: Enabled]
    end

    style Feature_Toggles fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Content_Settings fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Media_Settings fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

# Article Generation Settings Overview

```mermaid
graph TD
    subgraph Content_Settings
        A[Article Type] --> B[General Knowledge<br/>Single Section per Paragraph]
        C[Voice Tone] --> D[Professional]
        E[Language] --> F[English]
        G[Point of View] --> H[Default - Auto Select]
    end
    
    subgraph Feature_Toggles
        I[Enabled Features] --> J[Summary]
        I --> K[FAQ/PAA]
        I --> L[Images]
        I --> M[YouTube Videos]
        I --> N[External Links]
        
        O[Disabled Features] --> P[Grammar Check]
        O --> Q[Text Humanization]
    end
    
    subgraph Image_Config
        R[Image Settings] --> S[Min: 4 Images]
        R --> T[Max: 7 Images]
        R --> U[Landscape Orientation]
        R --> V[Relevant Sorting]
    end
    
    subgraph Token_Management
        W[Token Limits] --> X[Title: 100]
        W --> Y[Outline: 500]
        W --> Z[Intro: 800]
        W --> AA[Section: 2000]
        W --> AB[Conclusion: 800]
        W --> AC[PAA: 1000]
    end

    style Content_Settings fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Feature_Toggles fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Image_Config fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Token_Management fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

# Article Generation System Flow Documentation

## System Architecture Overview

```mermaid
graph TD
    subgraph Main_Script
        M[main.py]
    end

    subgraph Config
        C[config.py]
    end

    subgraph API_Validation
        AV[api_validators.py]
    end

    subgraph Article_Generator_Package
        AG[article_generator/__init__.py]
        CG[content_generator.py]
        TP[text_processor.py]
        IH[image_handler.py]
        WH[wordpress_handler.py]
        PH[paa_handler.py]
    end

    subgraph Utils_Package
        U[utils/__init__.py]
        TU[text_utils.py]
        FU[file_utils.py]
    end

    M --> C
    M --> AV
    M --> AG
    AG --> CG
    AG --> TP
    AG --> IH
    AG --> WH
    AG --> PH
    IH --> FU
    TP --> TU
```

## Main Script Flow

```mermaid
flowchart TD
    Start([Start]) --> Setup[Setup Environment]
    Setup --> ReadKeywords[Read Keywords File]
    ReadKeywords --> ProcessLoop{For each keyword}
    
    ProcessLoop --> |Next| GenerateArticle[Generate Article]
    GenerateArticle --> AddFAQ{Add FAQ?}
    AddFAQ --> |Yes| GenerateFAQ[Generate FAQ Section]
    AddFAQ --> |No| ProcessText
    GenerateFAQ --> ProcessText
    
    ProcessText --> CheckGrammar{Check Grammar?}
    CheckGrammar --> |Yes| FixGrammar[Fix Grammar]
    CheckGrammar --> |No| Humanize
    FixGrammar --> Humanize
    
    Humanize --> HumanizeText{Humanize?}
    HumanizeText --> |Yes| MakeHuman[Make Text Natural]
    HumanizeText --> |No| Format
    MakeHuman --> Format
    
    Format[Format for WordPress] --> Images{Add Images?}
    Images --> |Yes| GetImages[Get Images]
    Images --> |No| Publish
    GetImages --> Publish
    
    Publish[Post to WordPress] --> Success{Success?}
    Success --> |Yes| Continue
    Success --> |No| LogError[Log Error]
    
    Continue --> ProcessLoop
    LogError --> ProcessLoop
    
    ProcessLoop --> |Done| End([End])
```

## Content Generation Flow

```mermaid
sequenceDiagram
    participant M as Main
    participant CG as ContentGenerator
    participant TP as TextProcessor
    participant IH as ImageHandler
    participant WH as WordPressHandler
    participant PAA as PAAHandler

    M->>CG: generate_complete_article(keyword)
    activate CG
    CG->>CG: generate_title()
    CG->>CG: generate_outline()
    CG->>CG: generate_introduction()
    CG->>CG: generate_sections()
    CG->>CG: generate_conclusion()
    CG-->>M: article_data
    deactivate CG

    M->>PAA: generate_paa_section(keyword)
    activate PAA
    PAA->>PAA: get_paa_questions()
    PAA->>PAA: generate_paa_answer()
    PAA-->>M: faq_section
    deactivate PAA

    M->>TP: process_text(article)
    activate TP
    TP->>TP: check_grammar()
    TP->>TP: humanize_text()
    TP->>TP: format_article_for_wordpress()
    TP-->>M: formatted_article
    deactivate TP

    M->>IH: get_article_images(keyword)
    activate IH
    IH->>IH: get_image_list()
    IH->>IH: process_feature_image()
    IH->>IH: process_body_images()
    IH-->>M: images
    deactivate IH

    M->>WH: post_to_wordpress(article, images)
    activate WH
    WH->>WH: upload_media()
    WH->>WH: create_post()
    WH-->>M: post_data
    deactivate WH
```

## API Validation Flow

```mermaid
graph TD
    subgraph API_Validation_Process
        Start([Start]) --> ValidateOpenAI[Validate OpenAI API]
        ValidateOpenAI --> |Success| ValidateYT[Validate YouTube API]
        ValidateOpenAI --> |Fail| ErrorO[Log OpenAI Error]
        
        ValidateYT --> |Success| ValidateSerp[Validate SerpAPI]
        ValidateYT --> |Fail| ErrorY[Log YouTube Error]
        
        ValidateSerp --> |Success| ValidateWP[Validate WordPress]
        ValidateSerp --> |Fail| ErrorS[Log SerpAPI Error]
        
        ValidateWP --> |Success| Success[All APIs Valid]
        ValidateWP --> |Fail| ErrorW[Log WordPress Error]
        
        Success --> End([End])
        ErrorO --> End
        ErrorY --> End
        ErrorS --> End
        ErrorW --> End
    end
```

## Text Processing Flow

```mermaid
graph LR
    subgraph Text_Processing
        Input[Raw Text] --> Split[Split into Sentences]
        Split --> Distribute[Distribute Sentences]
        Distribute --> Grammar[Grammar Check]
        Grammar --> Humanize[Humanize Text]
        Humanize --> Format[Format for WordPress]
        Format --> Output[Formatted Text]
    end
```

## Image Handling Flow

```mermaid
graph TD
    subgraph Image_Processing
        Start([Start]) --> Search[Search Images]
        Search --> Feature[Process Feature Image]
        Search --> Body[Process Body Images]
        
        Feature --> Download1[Download Feature]
        Body --> Download2[Download Body Images]
        
        Download1 --> Optimize1[Optimize Feature]
        Download2 --> Optimize2[Optimize Body]
        
        Optimize1 --> WP1[Upload to WordPress]
        Optimize2 --> WP2[Upload to WordPress]
        
        WP1 --> End([End])
        WP2 --> End
    end
```

## WordPress Integration Flow

```mermaid
sequenceDiagram
    participant M as Main
    participant WH as WordPressHandler
    participant WP as WordPress

    M->>WH: post_to_wordpress(article, images)
    activate WH
    
    WH->>WH: get_credentials()
    
    loop For each image
        WH->>WP: upload_media()
        WP-->>WH: media_data
        WH->>WH: process_media_data()
    end
    
    WH->>WP: create_post()
    WP-->>WH: post_data
    
    WH-->>M: result
    deactivate WH
```

## Function Dependencies

```mermaid
graph TD
    subgraph Main_Functions
        main --> setup_environment
        main --> process_keyword
        process_keyword --> generate_complete_article
        process_keyword --> generate_paa_section
        process_keyword --> format_article_for_wordpress
        process_keyword --> get_article_images
        process_keyword --> post_to_wordpress
    end

    subgraph Helper_Functions
        generate_complete_article --> generate_title
        generate_complete_article --> generate_outline
        generate_complete_article --> generate_introduction
        generate_complete_article --> generate_paragraph
        generate_complete_article --> generate_conclusion
        
        format_article_for_wordpress --> wrap_with_paragraph_tag
        
        get_article_images --> get_image_list
        get_article_images --> process_feature_image
        get_article_images --> process_body_image
        
        post_to_wordpress --> upload_media_to_wordpress
        post_to_wordpress --> create_wordpress_post
    end
```

## Error Handling Flow

```mermaid
graph TD
    subgraph Error_Handling
        Error[Error Occurs] --> Type{Error Type}
        
        Type --> |API| APIError[API Error Handler]
        Type --> |File| FileError[File Error Handler]
        Type --> |Network| NetworkError[Network Error Handler]
        
        APIError --> Retry{Retry?}
        FileError --> Log[Log Error]
        NetworkError --> Retry
        
        Retry --> |Yes| Wait[Wait with Backoff]
        Retry --> |No| Log
        
        Wait --> Attempt[Retry Operation]
        Attempt --> Check{Success?}
        
        Check --> |Yes| Continue[Continue Operation]
        Check --> |No| Retry
        
        Log --> Report[Report Error]
    end
```

This documentation provides a comprehensive view of how the different components of the article generation system interact with each other. The Mermaid.js diagrams help visualize:

1. Overall system architecture
2. Main script execution flow
3. Content generation process
4. API validation process
5. Text processing pipeline
6. Image handling workflow
7. WordPress integration
8. Function dependencies
9. Error handling mechanisms

Each diagram represents a different aspect of the system, making it easier to understand the flow of data and control between different modules and functions. 

# Article Generation Flow Analysis

## Process Flow with Weakness Points

```mermaid
graph TD
    %% Main Process Flow
    Start([Start]) --> Init[Initialize ArticleContext]
    Init --> GenTitle[Generate Title]
    GenTitle --> GenOutline[Generate Outline]
    GenOutline --> GenIntro[Generate Introduction]
    GenIntro --> GenSections[Generate Sections]
    GenSections --> GenConc[Generate Conclusion]
    GenConc --> Format[Format Article]
    Format --> End([End])

    %% Weakness Points
    W1[Repetition Issue]
    W2[Citation Limitation]
    W3[Generic Examples]
    W4[Depth Inconsistency]

    %% Context Management Subgraph
    subgraph Context_Management
        TokenTrack[Token Tracking]
        ContextPrune[Context Pruning]
        HistoryMgmt[History Management]
        
        TokenTrack --> ContextPrune
        ContextPrune --> HistoryMgmt
    end

    %% Token Management Impact
    TokenTrack -.->|Causes| W1
    TokenTrack -.->|Affects| W4

    %% Model Limitations
    ModelLimit[GPT-4o-mini Limitations]
    ModelLimit -.->|Causes| W2
    ModelLimit -.->|Results in| W3

    %% Specific Components
    subgraph Content_Generation
        OutlineGen[Outline Generation]
        SectionGen[Section Generation]
        ContentRefine[Content Refinement]
    end

    %% Component Connections
    OutlineGen --> SectionGen
    SectionGen --> ContentRefine

    %% Impact on Quality
    ContentRefine -.->|Attempts to fix| W1
    SectionGen -.->|Contributes to| W4

    %% Token Window Limitation
    TokenWindow[4k Token Window]
    TokenWindow -.->|Restricts| Context_Management
    TokenWindow -.->|Limits| ContentRefine

    %% Strength Points
    S1[Coherent Narrative]
    S2[Well-Structured]
    S3[Engaging Intro]
    S4[Comprehensive]

    %% Positive Impact
    Context_Management -->|Enables| S1
    OutlineGen -->|Creates| S2
    ContentRefine -->|Produces| S3
    SectionGen -->|Achieves| S4

    classDef weakness fill:#ffebee,stroke:#c62828,stroke-width:2px;
    classDef strength fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef process fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef limitation fill:#fff3e0,stroke:#f57c00,stroke-width:2px;

    class W1,W2,W3,W4 weakness;
    class S1,S2,S3,S4 strength;
    class TokenTrack,ContextPrune,HistoryMgmt,OutlineGen,SectionGen,ContentRefine process;
    class ModelLimit,TokenWindow limitation;
```

## Analysis of Key Components

### 1. Context Management
- **ArticleContext Class**
  - Manages conversation history
  - Tracks token usage
  - Prunes context when needed
  - Impact on quality:
    * Enables coherent narrative
    * But limited by token window

### 2. Content Generation
- **Outline Generation**
  - Uses temperature=0.7 for creativity
  - Structures article hierarchy
  - Impact on quality:
    * Creates well-organized content
    * But can lead to depth inconsistency

### 3. Token Management
- **Token Window Limitation**
  - 4k token context window
  - Forces context pruning
  - Impact on quality:
    * Causes some repetition
    * Affects content depth

### 4. Model Limitations
- **GPT-4o-mini Constraints**
  - Knowledge cutoff
  - Training data limitations
  - Impact on quality:
    * Generic examples
    * Limited citations

## Improvement Opportunities

1. **Context Management**
   ```python
   def prune_context(self, tokens_needed: int):
       # Current: Removes oldest messages
       # Could: Implement smart pruning based on relevance
   ```

2. **Content Generation**
   ```python
   def generate_section(self, heading: str):
       # Current: Fixed temperature
       # Could: Adaptive temperature based on section type
   ```

3. **Token Management**
   ```python
   def manage_tokens(self):
       # Current: Simple threshold-based
       # Could: Implement sliding window with overlap
   ```

4. **Quality Control**
   ```python
   def validate_content(self):
       # Current: Basic validation
       # Could: Add fact-checking and citation validation
   ```

## Recommendations

1. **Short-term Fixes**
   - Implement smarter context pruning
   - Add section-specific temperature control
   - Enhance token management strategy

2. **Long-term Improvements**
   - Upgrade to larger model when available
   - Add fact-checking capabilities
   - Implement citation management
   - Develop example database 