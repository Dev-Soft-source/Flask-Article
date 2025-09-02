# Generator.py Refactoring Blueprint

## Current Issues
1. 1138 LOC - Too large and complex
2. Mixes core generation, helper utils, and WordPress handling
3. Deep nesting and complex flows

## Recommended Splits

### 1. Generation Orchestrator (new file)
```python
class GenerationOrchestrator:
    def run_full_generation(self, keyword):
        # Main workflow coordination
        # Calls other components in sequence
```

### 2. Component Factory (new file)
```python
class ComponentFactory:
    def create_title(self, context):
    
    def create_outline(self, context):

    def create_section(self, section_data):
```

### 3. RAG Integration Service (new file)
```python
class RAGService:
    def get_web_context(self, keyword):
    
    def process_for_generation(self, content):
```

### 4. Output Handlers (new file)
```python
class MarkdownHandler:
    def save_article(self, article_dict):

class WordPressHandler:
    def upload_article(self, article_dict):
```

### 5. Error Handling Service (new file)
```python
class GenerationErrorHandler:
    def handle_generation_error(self, error):
    
    def retry_failed_component(self, component):
```

## Implementation Steps

### Phase 1 (Extract Core Logic)
1. Move generation methods to ComponentFactory
2. Extract RAG logic to RAGService
3. Create ErrorHandler

### Phase 2 (Simplify Main Class)
1. Reduce Generator class to orchestration only
2. Delegate to new services
3. Remove helper methods

### Expected Benefits
- Clear separation of concerns
- Each file <400 lines
- Easier testing
- Better maintainability