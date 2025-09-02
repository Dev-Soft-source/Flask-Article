# Paragraph Headings Feature

## Overview

The paragraph headings feature enhances article readability by adding descriptive headings to individual paragraphs. This creates a more scannable, SEO-friendly article structure that helps readers quickly find relevant information.

## Configuration Settings

The following configuration parameters control paragraph headings:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_paragraph_headings` | bool | True | Enable or disable paragraph headings |
| `max_paragraph_headings_per_section` | int | 5 | Maximum number of paragraph headings per section |
| `refine_paragraph_headings` | bool | True | Allow the LLM to refine outline-based headings |
| `variable_paragraph_headings` | bool | False | Use a variable number of headings based on content |

## Usage Examples

### Basic Usage

To enable paragraph headings in your article generation:

```python
# In your configuration file
enable_paragraph_headings = True
```

### Disabling Paragraph Headings

If you prefer traditional paragraph formatting without headings:

```python
# In your configuration file
enable_paragraph_headings = False
```

### Advanced Configuration

For more control over paragraph headings:

```python
# In your configuration file
enable_paragraph_headings = True
max_paragraph_headings_per_section = 3  # Limit to 3 paragraph headings per section
refine_paragraph_headings = True  # Allow the LLM to refine outline-based headings
variable_paragraph_headings = True  # Use a variable number of headings based on content
```

## HTML Output Example

When paragraph headings are enabled, the HTML output will include `<h4>` tags for paragraph headings:

```html
<h2>Main Section Heading</h2>

<h4>First Paragraph Heading</h4>
<p>This is the content of the first paragraph with its own descriptive heading...</p>

<h4>Second Paragraph Heading</h4>
<p>This is the content of the second paragraph with its own descriptive heading...</p>
```

## Markdown Output Example

For Markdown format, paragraph headings are rendered as level 4 headings (####):

```markdown
## Main Section Heading

#### First Paragraph Heading

This is the content of the first paragraph with its own descriptive heading...

#### Second Paragraph Heading

This is the content of the second paragraph with its own descriptive heading...
```

## Implementation Details

Paragraph headings are generated using a single API call that produces both the heading and paragraph content. The response is parsed using the following format:

```
[HEADING] Brief, descriptive heading for this paragraph
[CONTENT] The paragraph content...
```

This approach ensures consistency between the heading and content while minimizing API calls.

## Fallback Mechanisms

If paragraph heading generation fails:

1. The system will attempt to extract heading and content using regex patterns
2. If parsing fails, a generic heading ("Additional Information") will be used
3. If WordPress formatting doesn't find heading/paragraph pairs, it will fall back to standard paragraph processing

## Best Practices

- Use paragraph headings for longer articles (1500+ words)
- Set `max_paragraph_headings_per_section` based on section complexity
- For technical content, enable `refine_paragraph_headings` for more accurate headings
- Test both with headings enabled and disabled to determine which works better for your content
