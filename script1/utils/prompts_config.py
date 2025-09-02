"""Configuration class for article generation prompts with HTML formatting rules."""

from dataclasses import dataclass
import textwrap


@dataclass
class Prompts:
    """Dataclass containing all prompts used in article generation."""

    # HTML formatting guidelines
    HTML_GUIDELINES: str = textwrap.dedent("""
        Follow these HTML formatting rules:
        - Bold important keywords with <strong>keyword</strong>
        - Use <em>text</em> for emphasis
        - Create bullet lists with <ul><li>item</li></ul>
        - Create numbered lists with <ol><li>step</li></ol>
        - Add tables with proper <table><thead><tbody> structure
        - Keep all formatting clean and WordPress-compatible""")

    # Reusable phrases
    LANG_TONE_POV: str = "Use {articlelanguage} with a {voicetone} tone and {pointofview} perspective."

    # Core article structure prompts
    title: str = f"""Generate a compelling {{articletype}} article title targeting {{articleaudience}} readers about {{keyword}}. 
Write in {{articlelanguage}} with a {{voicetone}} tone. Keep it clear, engaging, SEO-friendly, and between 50-60 characters.
{HTML_GUIDELINES}"""

    title_craft: str = f"""You are a skilled SEO copywriter optimizing article titles. Enhance this title while keeping the core keyword and meaning.
{HTML_GUIDELINES}"""

    fixed_outline: str = f"""Create a detailed outline for a {{articletype}} article about {{keyword}} aimed at {{articleaudience}} readers.
{LANG_TONE_POV}
Include {{size_headings}} main sections with {{size_sections}} subsections each.
Structure the outline using Roman numerals (I., II., etc.) for main sections and letters (A., B., etc.) for subsections.
Ensure each section builds logically and maintains engagement.
{HTML_GUIDELINES}"""

    variable_paragraphs_outline: str = f"""Create a {{sizesections}}-section outline about {{keyword}} for {{articleaudience}} readers.
Focus on practical, actionable content. Use Roman numerals (I, II, III) for main sections and letters (A, B, C) for {{sizeheadings}} subsections.
For each subsection, determine the optimal number of paragraphs (1-5) based on content complexity.
Aim for subsections that work well as tables or lists.
Return only the outline in <outline> tags.
{HTML_GUIDELINES}"""

    two_level_outline: str = f"""Create a {{sizesections}}-section outline about {{keyword}} for {{articleaudience}} readers.
Focus on practical, actionable content. Use Roman numerals (I, II, III) for main sections.
For each main section, provide {{sizeparagraphs}} paragraph points (numbered 1, 2, 3...) that will become individual paragraphs.
Aim for content that works well as tables or lists.
Return only the outline in <outline> tags.
{HTML_GUIDELINES}"""

    variable_two_level_outline: str = f"""Create a {{sizesections}}-section outline about {{keyword}} for {{articleaudience}} readers.
Focus on practical, actionable content. Use Roman numerals (I, II, III) for main sections.
For each main section, determine the optimal number of paragraph points (1-5) based on content complexity.
Aim for content that works well as tables or lists.
Return only the outline in <outline> tags.
{HTML_GUIDELINES}"""

    introduction: str = f"""Write an engaging introduction for a {{articletype}} article about {{keyword}} for {{articleaudience}} readers.
{LANG_TONE_POV}
Hook the reader immediately, establish context, and preview the value they'll get from reading.
Adapt the complexity and terminology to the audience's expertise level.
{HTML_GUIDELINES}"""

    paragraph_generate: str = f"""Write section {{section_number}} of {{total_sections}} about "{{heading}}" for a {{articletype}} article on {{keyword}}.
Target {{articleaudience}} readers. {LANG_TONE_POV}
This is paragraph {{current_paragraph}} of {{paragraphs_per_section}}.
Maintain a length between {{min_paragraph_tokens}} and {{max_paragraph_tokens}} tokens.
Adjust technical depth and examples based on audience knowledge.
Represent information using <strong>, <em>, <ul>, <ol>, and tables where appropriate.
{HTML_GUIDELINES}"""

    conclusion: str = f"""Write a conclusion for a {{articletype}} article about {{keyword}} targeting {{articleaudience}} readers.
{LANG_TONE_POV}
Summarize key points, reinforce the main message, and include a relevant call to action.
Ensure the conclusion resonates with the audience.
Use HTML formatting as per guidelines.
{HTML_GUIDELINES}"""

    # Enhancement prompts
    humanize: str = f"""Make this content more engaging and natural while maintaining its core message.
Keep it conversational and engaging for {{articleaudience}} audiences, or formal/precise for professional topics.
Use emotional cues sparingly for technical content.
{HTML_GUIDELINES}"""

    summarize: str = f"""Create a concise summary of the article that captures its essence for {{articleaudience}} readers.
Highlight the most relevant points while maintaining clarity and engagement.
Format using HTML rules: <strong>, <em>, <ul>, <ol>, <table>.
{HTML_GUIDELINES}"""

    faqs: str = f"""Generate relevant FAQs about {{keyword}} that would interest {{articleaudience}} readers.
Provide clear, actionable answers at the appropriate technical level.
Use HTML formatting for lists, emphasis, and tables.
{HTML_GUIDELINES}"""

    paa_answer: str = f"""You are writing an answer for a "People Also Ask" section about "{{keyword}}".
The question is: "{{question}}"

Write a concise, informative paragraph that directly answers the question.
Keep the answer to a single paragraph (~100-150 words).
Use HTML formatting for <strong>, <em>, lists, and tables.
{HTML_GUIDELINES}"""

    blocknote: str = f"""Create engaging block notes highlighting key points about {{keyword}} for {{articleaudience}} readers.
Focus on information this audience finds most valuable.
Format using HTML <strong>, <em>, <ul>, <ol>, and tables.
{HTML_GUIDELINES}"""

    grammar: str = f"""Check and improve the grammar while maintaining the appropriate tone and complexity for {{articleaudience}} readers.
Preserve HTML formatting <strong>, <em>, <ul>, <ol>, <table>.
{HTML_GUIDELINES}"""

    meta_description: str = f"""Write an SEO-optimized meta description about {{keyword}} for {{articleaudience}} readers.
Use {LANG_TONE_POV} and keep it under 155 characters.
{HTML_GUIDELINES}"""

    wordpress_excerpt: str = f"""Create an engaging WordPress excerpt about {{keyword}} for {{articleaudience}} readers.
Use {LANG_TONE_POV}. Make it informative and compelling.
Follow HTML formatting rules.
{HTML_GUIDELINES}"""

    summary_combine: str = f"""Combine multiple chunks of article summary into a single coherent result.
Eliminate repetition while maintaining all unique information.
Use HTML formatting for emphasis, lists, and tables.
{HTML_GUIDELINES}"""

    blocknotes_combine: str = textwrap.dedent(f"""\
        Combine multiple chunks of key takeaways into a single coherent paragraph following these strict requirements:
        1. Length: Max 150 words
        2. Content: Only the most critical and actionable insights
        3. Structure: Logical flow
        4. Style: Authoritative and practical
        5. Format: HTML only (<strong>, <em>, <ul>, <ol>, <table>)
        6. Include: 5-6 essential points
        7. Avoid meta-commentary (e.g., "In conclusion")
        Chunks to combine:
        {{chunks_text}}
        RESPONSE FORMAT:
        [Begin with the combined key takeaways directly, no introduction]
        {HTML_GUIDELINES}""")

    # ----------------------
    # Post-init validation
    # ----------------------
    def __post_init__(self):
        """Validate that all required prompts are present and non-empty."""
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            if not value or not str(value).strip():
                raise ValueError(f"Missing or empty prompt: {field_name}")

    # ----------------------
    # Safe formatting helper
    # ----------------------
    def render(self, field_name: str, **kwargs) -> str:
        """Safely format a prompt by field name."""
        template = getattr(self, field_name, None)
        if template is None:
            raise ValueError(f"Prompt '{field_name}' not found.")
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing placeholder for {e.args[0]} in '{field_name}' prompt.")
