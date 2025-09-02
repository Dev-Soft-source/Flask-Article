# Script2 Generator Refactoring Plan

## Current Issues
- 1567 LOC - Violates SRP
- Mixes:
  - Article orchestration
  - Component generation 
  - API validations
  - Error handling
  - Output processing

## Recommended File Structure

### 1. article_orchestrator.py (New)
```python
class ArticleOrchestrator:
    """Main workflow controller"""
    
    def generate_article(self, keyword):
        """Top-level generation workflow"""
    
    def _generate_components(self):
        """Coordinate component creation"""

    def _assemble_article(self):
        """Combine all components"""
```

### 2. api_validator_service.py (New)
```python
class APIValidatorService:
    """Dedicated API validation"""
    
    def validate_all(self):
        """Validate all required APIs"""
    
    def validate_openai(self):
        """OpenAI specific checks"""

    # Other API-specific methods...
```

### 3. component_factory.py (New)
```python
class ComponentFactory:
    """Creates individual article components"""
    
    def create_title(self, context):
    
    def create_sections(self, headings):
    
    def create_metadata(self):
    
    # Other component creation methods...
```

### 4. output_handler.py (New)
```python
class OutputHandler:
    """Handles final output generation"""
    
    def save_markdown(self, article):
    
    def publish_wordpress(self, article):
    
    def format_output(self):
```

## Implementation Steps

### Phase 1 - Separation
1. Extract API validation to dedicated service (400 → 150 lines)
2. Move output handling to new class (300 → 200 lines)
3. Isolate component generation (600 → 400 lines)

### Phase 2 - Refinement 
1. Implement proper dependency injection
2. Add intermediate validation checks
3. Standardize error handling

## Benefits
1. Each file under 400 lines
2. Clear responsibility boundaries
3. Improved testability
4. Better error isolation
5. Easier maintenance