#  **بِسْمِ اللهِ الرَّحْمٰنِ الرَّحِيْمِ**


# Flexible CSV Parser Implementation Plan

**Date Created**: May 26, 2025  
**Status**: Planning Phase  
**Goal**: Implement flexibility in CSV parser for dynamic article generation with varying subtitle counts

## Overview

The current CSV parser in script2 expects all articles to have the same number of subtitles. This implementation will create a flexible system that allows articles to have varying numbers of subtitles (1-20) while maintaining backward compatibility.

## Problem Statement

### Current Limitations
- **Rigid Column Structure**: All articles must have identical subtitle count
- **Fixed Mapping**: Hardcoded subtitle/image column pairs (subtitle1-subtitle12, img1-img12)
- **Strict Validation**: Only processes subtitle/image pairs if both columns exist and subtitle is not empty
- **No Dynamic Detection**: Cannot adapt to varying article structures within same CSV

### Target Solution
- Dynamic detection of subtitle/image columns per row
- Variable number of subtitles per article (1-20)
- Backward compatibility with existing CSV formats
- Intelligent column pairing and flexible validation

## Implementation Plan

### Phase 1: Core Dynamic Detection ✅
**Status**: Completed

#### Files to Modify:
- `/scripts/utils/unified_csv_processor.py`

#### New Methods to Implement:

1. **`_detect_dynamic_subtitle_columns(self, headers: List[str]) -> Tuple[List[str], List[str]]`**
   - [x] Scan headers for subtitle patterns (`subtitle1`, `subtitle2`, `sub1`, `sub2`, etc.)
   - [x] Scan headers for image patterns (`img1`, `image1`, `pic1`, etc.)
   - [x] Return matched pairs of subtitle and image columns
   - [x] Handle edge cases and malformed patterns

2. **`_get_flexible_column_mapping(self, headers: List[str]) -> Dict[str, Any]`**
   - [x] Detect required columns (keyword, title, etc.)
   - [x] Detect optional columns (featured_img, etc.)
   - [x] Detect dynamic subtitle/image columns
   - [x] Create comprehensive mapping dictionary

#### Properties to Add:
- [x] `dynamic_subtitle_columns` property
- [x] `dynamic_image_columns` property
- [x] Update class initialization for dynamic column storage

### Phase 2: Flexible Processing ✅
**Status**: Completed

#### Methods to Implement:

1. **`_extract_subtitle_image_pairs(self, row: Dict[str, str], subtitle_cols: List[str], image_cols: List[str]) -> List[Dict[str, str]]`**
   - [x] Iterate through detected subtitle columns
   - [x] Extract non-empty subtitles with corresponding images
   - [x] Handle missing image columns gracefully
   - [x] Skip empty subtitles but continue processing remaining ones
   - [x] Return structured list of subtitle/image pairs

#### Methods to Modify:

1. **`process_structured_file(self)` (lines ~260-330)**
   - [x] Replace hardcoded column lists with dynamic detection
   - [x] Integrate `_detect_dynamic_subtitle_columns()` call
   - [x] Use `_extract_subtitle_image_pairs()` for processing
   - [x] Maintain backward compatibility with existing formats
   - [x] Add comprehensive error handling

2. **`validate_file(self)` (lines ~400-500)**
   - [x] Make validation more permissive for subtitle/image columns
   - [x] Require only core columns (keyword, title)
   - [x] Allow optional subtitle/image columns with warnings
   - [x] Add validation for dynamic column detection results
   - [x] Provide informative feedback for detected columns

3. **`_is_structured_csv(self, lines: List[str])` (lines ~130-150)**
   - [x] Enhance detection for dynamic subtitle patterns
   - [x] Look for numbered column patterns
   - [x] Consider CSV structured if core columns + subtitle patterns found
   - [x] Improve pattern recognition accuracy

### Phase 3: Configuration & Integration ✅
**Status**: Completed

#### Files to Modify:

1. **`/scripts/script2/config.py`**
   - [x] Add `CSV_FLEXIBLE_PARSING = True` flag
   - [x] Add `CSV_MAX_SUBTITLES = 20` setting
   - [x] Add `CSV_SUBTITLE_PATTERNS = ['subtitle', 'sub', 'heading', 'section']`
   - [x] Add `CSV_IMAGE_PATTERNS = ['img', 'image', 'pic', 'photo']`
   - [x] Maintain existing column definitions as fallback
   - [x] Add configuration validation

2. **`/scripts/script2/utils/csv_utils.py`**
   - [x] Update `process_csv_file()` to pass flexible parsing flag
   - [x] Handle dynamic column detection results
   - [x] Ensure compatibility with existing article generation pipeline
   - [x] Add logging for flexible parsing operations

### Phase 4: Testing & Validation ✅
**Status**: Completed

#### Test Files to Create:

1. **`/scripts/script2/test_flexible_csv.py`**
   - [x] Test CSV with articles having 2, 3, 5 subtitles respectively
   - [x] Test CSV with missing image columns for some subtitles
   - [x] Test CSV with mixed column naming patterns
   - [x] Test backward compatibility with existing CSV formats
   - [x] Test edge cases and error conditions
   - [x] Performance testing with large CSVs

2. **`/scripts/script2/test_flexible_input.csv`**
   - [x] Create sample CSV with varying subtitle counts
   - [x] Include mixed image availability scenarios
   - [x] Demonstrate flexible parsing capabilities
   - [x] Provide comprehensive test data

#### Test Scenarios:
- [x] **Scenario 1**: CSV with 2-5 subtitles per article
- [x] **Scenario 2**: CSV with missing image columns
- [x] **Scenario 3**: CSV with mixed naming patterns
- [x] **Scenario 4**: Backward compatibility test
- [x] **Scenario 5**: Large CSV performance test
- [x] **Scenario 6**: Error handling and edge cases

## Technical Specifications

### Dynamic Column Detection Logic
```
Subtitle Patterns: subtitle1, subtitle2, sub1, sub2, heading1, section1
Image Patterns: img1, img2, image1, pic1, photo1
Max Subtitles: 20 (configurable)
Pairing Logic: subtitle{n} ↔ img{n}/image{n}/pic{n}
```

### Backward Compatibility Strategy
1. **Default Behavior**: Fall back to config-defined columns if no dynamic columns detected
2. **Legacy CSV Support**: Existing CSV files continue to work unchanged
3. **Configuration Flags**: Allow disabling flexible parsing if needed
4. **Error Handling**: Graceful degradation when dynamic detection fails

### Expected Data Structure
```python
# Per article output structure
{
    "keyword": "example keyword",
    "title": "Article Title",
    "featured_img": "featured_image_url",
    "subtitle_image_pairs": [
        {"subtitle": "First Subtitle", "image": "image1_url"},
        {"subtitle": "Second Subtitle", "image": "image2_url"},
        {"subtitle": "Third Subtitle", "image": None}  # No image for this subtitle
    ]
}
```

## Progress Tracking

### Milestones
- [x] **Milestone 1**: Dynamic detection methods implemented
- [x] **Milestone 2**: Flexible processing logic integrated
- [x] **Milestone 3**: Configuration and integration complete
- [x] **Milestone 4**: Comprehensive testing completed
- [x] **Milestone 5**: Documentation and deployment ready

### Current Status
- ✅ Problem analysis completed
- ✅ Implementation plan created
- ✅ Implementation completed
- ✅ Testing completed
- ✅ Documentation completed

## Benefits Expected

### For Content Creators
- **Freedom**: Create articles with varying subtitle structures
- **Efficiency**: No need to fill empty subtitle columns
- **Flexibility**: Adapt article structure to content requirements

### For System
- **Scalability**: Handle diverse content requirements
- **Maintainability**: Less hardcoded logic, more adaptive processing
- **Performance**: Process only actual content, skip empty columns
- **Robustness**: Better error handling and validation

## Risk Mitigation

### Technical Risks
- [ ] **Risk 1**: Dynamic detection accuracy - *Mitigation: Comprehensive pattern testing*
- [ ] **Risk 2**: Backward compatibility issues - *Mitigation: Fallback mechanisms*
- [ ] **Risk 3**: Performance degradation - *Mitigation: Benchmarking and optimization*
- [ ] **Risk 4**: Integration conflicts - *Mitigation: Staged implementation and testing*

### Implementation Safeguards
- Comprehensive test suite
- Fallback to existing logic on detection failure
- Detailed logging for debugging
- Validation of processed data structure
- Performance monitoring

## Notes and Considerations

### Development Notes
- Maintain existing code style and patterns
- Follow existing error handling conventions
- Use rich console for user feedback
- Implement comprehensive logging

### Future Enhancements
- Support for custom column naming patterns
- GUI for CSV column mapping
- Advanced validation rules
- Export/import of column mappings

---

**Last Updated**: May 28, 2025  
**Next Review**: After deployment and usage feedback  
**Implementation Lead**: GitHub Copilot  
