# Text Processors Refactoring Plan (script1 + script2)

## Current Issues
1. **Duplicated Functionality** (844 + 763 lines total)
   - Similar core capabilities in both files
   - Divergent implementations for script-specific needs
2. **Mixed Responsibilities**:
   - Text processing
   - Formatting (WordPress/HTML/Markdown)
   - API calls to language models
3. **Inconsistent Interfaces**  
   - Different function signatures for similar operations
4. **Brittle Markdown Processing**
   - Hardcoded regex patterns
   - Fragile handling of edge cases

## Unified Architecture Proposal

### 1. core_text_processor.py (New Base Class)
```python
class TextProcessorCore:
    """Shared foundation for all text processing"""
    
    def humanize_text(self, text: str, config: Config) -> str:
        """Base humanization implementation"""
    
    def check_grammar(self, text: str, config: Config) -> str:
        """Base grammar checking"""
    
    # Other shared methods...
```

### 2. wordpress_processor.py (New)
```python
class WordPressProcessor(TextProcessorCore):
    """WordPress-specific formatting"""
    
    def format_article(self, article: dict) -> str:
        """Convert article components to WordPress format"""
    
    def clean_wordpress_html(self, content: str) -> str:
        """Sanitize WordPress HTML output"""
```

### 3. markdown_processor.py (New)  
```python
class MarkdownProcessor(TextProcessorCore):
    """Markdown-specific processing"""
    
    def format_article(self, article: dict) -> str:
        """Convert article components to Markdown"""
        
    def clean_markdown(self, content: str) -> str:
        """Sanitize markdown output"""
```

### 4. conversion_service.py (New)
```python
class FormatConversionService:
    """Cross-format conversion handler"""
    
    def wp_to_markdown(self, content: str) -> str:
        """WordPress → Markdown"""
    
    def markdown_to_wp(self, content: str) -> str:   
        """Markdown → WordPress"""
```

## Implementation Steps

### Phase 1: Core Extraction
1. Identify and extract shared functionality into base class
2. Create standardized interfaces
3. Implement common utilities:
   - Sentence splitting  
   - Paragraph distribution
   - Format cleaning

### Phase 2: Specialization
1. Move WordPress-specific logic to WordPressProcessor
2. Move Markdown logic to MarkdownProcessor  
3. Implement consistent error handling

### Phase 3: Cross-Script Harmonization
1. Align script1 and script2 implementations
2. Create migration paths
3. Update documentation

## Benefits
1. **~40% Code Reduction** through shared base
2. **Clearer Responsibility Separation** (SRP)
3. **Easier Maintenance** with consistent interfaces
4. **Better Testability** of isolated components
5. **Smoother Format Conversion** paths

## Estimated Savings
| File                  | Current LOC | Projected LOC | Reduction |
|-----------------------|------------:|--------------:|----------:|
| script1/text_processor | 844         | 300           | 64%       |  
| script2/text_processor | 763         | 300           | 61%       |
| **Total**             | 1607        | 600           | 63%       |