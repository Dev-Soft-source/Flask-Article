# Content Generator Refactoring Plan

## Current Issues
- 1647 LOC - Too large and complex
- Mixes multiple responsibilities:
  - API communication
  - Content generation logic
  - Context management
  - Formatting/output handling

## Recommended Structure

### 1. api_communicator.py (New)
```python
class APINegotiator:
    """Handles all API communications"""
    
    def call_openai(self, messages, model, params):
        """Make OpenAI API call with retry logic"""

    def call_openrouter(self, messages, model, params):
        """Make OpenRouter API call with retry logic"""

    def get_best_provider(self, model):
        """Select optimal API provider based on model"""
```

### 2. generation_service.py (New)
```python
class ContentGenerationService:
    """Core content generation logic"""
    
    def generate_title(self, context):
        """Title generation implementation"""

    def generate_section(self, context):
        """Section generation implementation"""

    # Other content generation methods...
```

### 3. context_manager.py (Extract from current)
```python
class ArticleContextManager:
    """Improved context handling"""
    
    def manage_tokens(self, new_content):
        """Token usage management"""

    def save_context(self):
        """Context saving logic"""

    def load_context(self):
        """Context loading logic"""
```

### 4. output_formatter.py (New)
```python
class OutputFormatter:
    """Handles all output formatting"""
    
    def format_for_wordpress(self, content):
        """WordPress-specific formatting"""

    def format_for_markdown(self, content):
        """Markdown formatting"""

    def clean_html(self, content):
        """HTML sanitization"""
```

## Implementation Steps

### Phase 1 - Core Separation
1. Extract all API calls to APINegotiator
2. Move context management to ArticleContextManager
3. Isolate output formatting in OutputFormatter

### Phase 2 - Logic Reorganization
1. Break content generation into focused methods
2. Implement proper dependency injection
3. Add intermediate result validation

## Benefits
1. Each file <400 lines
2. Clear separation of concerns
3. Easier testing and maintenance
4. Better error isolation
5. Improved single responsibility