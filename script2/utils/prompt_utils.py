"""Utilities for prompt processing."""

def generate_example_outline(sizesections, sizeheadings):
    """
    Generate an example outline structure based on the configuration.
    
    Args:
        sizesections (int): Number of main sections to include
        sizeheadings (int): Number of subsections per main section
        
    Returns:
        str: A formatted example outline
    """
    roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", 
                      "XI", "XII", "XIII", "XIV", "XV"]
    
    # Limit to a reasonable number
    sizesections = min(sizesections, len(roman_numerals))
    sizeheadings = min(sizeheadings, 15)
    
    # Generate subsection letters (A through O)
    subsection_letters = [chr(65 + i) for i in range(sizeheadings)]
    
    outline = []
    
    for i in range(sizesections):
        # Add section header
        outline.append(f"{roman_numerals[i]}. [Main Section Title]")
        
        # Add subsections
        for j in range(sizeheadings):
            outline.append(f"{subsection_letters[j]}. [Subsection Point]")
        
        # Add blank line between sections
        if i < sizesections - 1:
            outline.append("")
    
    return "\n".join(outline)
