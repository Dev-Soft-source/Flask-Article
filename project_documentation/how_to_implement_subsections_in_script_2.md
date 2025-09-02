# How to Implement Subsections in Script 2 - Corrected Plan

## Understanding Script 2 Requirements

**Key Difference**: Script 2 uses CSV input where you manually provide main section titles, unlike Script 1 which generates everything automatically.

### Parameter Clarification:
- **size_sections**: **IRRELEVANT for Script 2** - Number of main sections is determined by CSV input
- **size_headings**: **CRITICAL** - Number of subsections per main section (0 = no subsections, 2-level hierarchy)
- **size_paragraphs**: Number of paragraph points to generate for each subsection (3-level) or main section (2-level)

### Current Issues to Fix First:
1. **Not using CSV section titles as base** - currently generating independent outlines
2. **Always generating 3-level hierarchy** - ignoring size_headings parameter
3. **Missing 2-level hierarchy support** - no prompts for when size_headings=0

## Corrected Implementation Plan

### Phase 1: Fix Basic Structure extraction works correctly(Priority 1)
- [ ] **Add TWO_LEVEL_OUTLINE_PROMPT** to script2/prompts_simplified.py
- [ ] **Fix outline generation** to use CSV section titles as the foundation
- [ ] **Implement conditional prompt selection** based on size_headings parameter

### Phase 2: Update Prompt System
**New Prompts Needed in script2/prompts_simplified.py:**

```python
# 2-level hierarchy prompt (when size_headings=0)
CSV_TWO_LEVEL_OUTLINE_PROMPT = """Generate paragraph points for these {csv_sections_count} main sections about {keyword}: {csv_sections}
Focus on practical, actionable content for {articleaudience} readers.
For each section, provide {sizeparagraphs} specific paragraph points (numbered 1, 2, 3...) that will become individual paragraphs.
Return only the outline in <outline> tags."""

# 3-level hierarchy prompt (when size_headings>0)  
CSV_THREE_LEVEL_OUTLINE_PROMPT = """Generate subsections and paragraph points for these {csv_sections_count} main sections about {keyword}: {csv_sections}
Focus on practical, actionable content for {articleaudience} readers.
For each main section, create {sizeheadings} subsections (labeled A, B, C...) and provide {sizeparagraphs} paragraph points (numbered 1, 2, 3...) for each subsection.
Return only the outline in <outline> tags."""
```

### Phase 3: Fix Outline Generation Logic
**Location**: script2/article_generator/content_generator.py - generate_outline() method

**Changes needed:**
1. Accept CSV section titles as parameter
2. Select appropriate prompt based on size_headings (not size_sections)
3. Generate points/subsections based on CSV titles, not new titles
4. Pass CSV section titles to prompt

### Phase 4: Fix Section Generation Logic
**Location**: script2/article_generator/generator.py - _generate_core_components() method

**Critical Change**: Always use CSV section titles as headings, regardless of outline generation

**Remove/Modify**: Lines 555-563 (fallback to outline for headings)

### Phase 5: Fix Parsing Logic
**Location**: script2/article_generator/generator.py - _generate_sections() method

**Current issue**: Lines 626-645 expect 3-level structure but need to handle both 2-level and 3-level

**New parsing logic needed:**
- **2-level**: Parse paragraph points directly under CSV section titles
- **3-level**: Parse subsections under CSV section titles, then paragraph points under subsections

### Phase 6: Update Formatting Functions
**Critical**: These functions need updates to handle new hierarchy structure

#### WordPress Formatting Updates:
**Location**: script2/formatting/wordpress_formatter.py
- [ ] Update `format_wordpress_content()` to handle 2-level vs 3-level hierarchy
- [ ] Ensure proper heading levels (h2 for CSV sections, h3 for subsections when size_headings>0)
- [ ] Update TOC generation for both hierarchy types
- [ ] Handle paragraph headings appropriately

#### Markdown Formatting Updates:
**Location**: script2/formatting/markdown_formatter.py  
- [ ] Update `format_markdown_content()` for new hierarchy
- [ ] Ensure proper heading markdown (## for CSV sections, ### for subsections when size_headings>0)
- [ ] Update TOC generation
- [ ] Handle paragraph formatting

### Phase 7: Update Grammar & Humanization Functions
**Location**: script2/article_generator/content_generator.py and related processing functions

**Updates needed:**
- [ ] Update grammar checking to handle new hierarchical structure
- [ ] Update humanization functions to maintain hierarchy integrity
- [ ] Ensure paragraph breaks and formatting are preserved correctly
- [ ] Update chunking logic if needed for large hierarchical content

### Phase 8: **CRITICAL - Chunking & Block Notes Review**
**Developer Note**: Please review the following functions for compatibility with new hierarchical structure:

#### Block Notes Generation:
**Location**: script2/article_generator/generator.py - _generate_block_notes() method
- [ ] **Review chunking algorithm** to ensure it respects section/subsection boundaries
- [ ] **Update block notes generation** to handle both 2-level and 3-level hierarchies
- [ ] **Ensure block notes don't break** between subsections when size_headings>0

#### Key Takeaways Generation:
**Location**: script2/article_generator/content_generator.py - generate_key_takeaways() method
- [ ] **Review key takeaways extraction** to work with new structure
- [ ] **Ensure takeaways are extracted** from correct hierarchical levels
- [ ] **Handle both hierarchy types** appropriately

#### Summary Generation:
**Location**: script2/article_generator/content_generator.py - generate_summary() method
- [ ] **Review summary generation** to handle hierarchical content
- [ ] **Ensure summaries respect** section/subsection boundaries
- [ ] **Update chunking for summary** to work with new structure

#### Chunking Utils:
**Location**: script2/article_generator/chunking_utils.py
- [ ] **Review chunk_article_for_processing()** for hierarchical compatibility
- [ ] **Update combine_chunk_results()** to maintain hierarchy
- [ ] **Test chunk boundaries** don't split subsections inappropriately

### Expected Final Behavior:

#### When size_headings = 0 (2-level hierarchy):
```
CSV Section Title 1 (h2)
- Paragraph Point 1 → becomes paragraph 1
- Paragraph Point 2 → becomes paragraph 2
- Paragraph Point 3 → becomes paragraph 3

CSV Section Title 2 (h2)
- Paragraph Point 1 → becomes paragraph 1
- Paragraph Point 2 → becomes paragraph 2
- Paragraph Point 3 → becomes paragraph 3
```

#### When size_headings > 0 (3-level hierarchy):
```
CSV Section Title 1 (h2)
A. Generated Subsection 1 (h3)
- Paragraph Point 1 → becomes paragraph 1
- Paragraph Point 2 → becomes paragraph 2
B. Generated Subsection 2 (h3)
- Paragraph Point 1 → becomes paragraph 1
- Paragraph Point 2 → becomes paragraph 2

CSV Section Title 2 (h2)
A. Generated Subsection 1 (h3)
- Paragraph Point 1 → becomes paragraph 1
- Paragraph Point 2 → becomes paragraph 2
```

### Implementation Order for Developer:
1. **Add new prompts** to script2/prompts_simplified.py
2. **Fix outline generation** to use CSV titles and size_headings parameter
3. **Fix section generation** to always use CSV titles as headings
4. **Fix parsing logic** in _generate_sections for both hierarchy types
5. **Update WordPress formatting** functions
6. **Update Markdown formatting** functions  
7. **Update grammar/humanization** functions
8. **CRITICAL: Review chunking & block notes** functions for hierarchical compatibility
9. **Test 2-level hierarchy** first (size_headings=0)
10. **Test 3-level hierarchy** (size_headings>0)
11. **Update documentation** and examples

### Testing Checklist:
- [ ] Test with size_headings=0 (should generate 2-level hierarchy)
- [ ] Test with size_headings=2,3,4 (should generate 3-level hierarchy)
- [ ] Verify CSV section titles are always used as main headings
- [ ] Verify WordPress formatting works for both hierarchy types
- [ ] Verify Markdown formatting works for both hierarchy types
- [ ] Verify grammar/humanization preserves hierarchy
- [ ] **Verify chunking doesn't break section/subsection boundaries**
- [ ] **Verify block notes generation works with new structure**
- [ ] **Verify key takeaways extraction works correctly**
- [ ] **Verify summary generation handles hierarchy properly**
- [ ] Verify no fallback to generic points when parsing works correctly