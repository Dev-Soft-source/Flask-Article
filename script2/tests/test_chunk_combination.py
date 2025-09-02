import unittest
from dataclasses import dataclass
from article_generator.chunking_utils import combine_chunk_results_with_llm
from prompts import SUMMARY_COMBINE_PROMPT, BLOCKNOTES_COMBINE_PROMPT


@dataclass
class MockConfig:
    enable_separate_summary_model: bool = False
    summary_combination_paragraphs: int = 2
    keynotes_combination_paragraphs: int = 1
    summary_max_tokens: int = 800
    keynotes_max_tokens: int = 300
    openai_model: str = "gpt-4o-mini-2024-07-18"
    enable_rate_limiting: bool = False
    token_limits = {
        "title": 100,
        "outline": 500,
        "introduction": 800,
        "section": 2000,
        "conclusion": 800,
        "paa": 1000,
        "faq": 1000,
        "summary_combination": 700,  # Default max tokens for combining summary chunks
        "keynotes_combination": 300,  # Default max tokens for combining blocknotes
    }
    use_openrouter = True


class MockContext:
    def __init__(self):
        self.config = MockConfig()
        self.articleaudience = "intermediate readers"
        self.articlelanguage = "English"
        self.voicetone = "professional"

    def add_message(self, role, content):
        # Mock method to simulate context.add_message
        pass


class TestChunkCombination(unittest.TestCase):
    def setUp(self):
        self.context = MockContext()

        # Test data
        self.summary_chunks = [
            "First chunk of the summary discusses important points A and B.",
            "Second chunk contains points C and D.",
            "Third chunk talks about conclusions E and F.",
        ]

        self.blocknote_chunks = [
            "<strong>Key point 1:</strong> Important insight about topic A.",
            "<strong>Key point 2:</strong> Critical finding about aspect B.",
            "<strong>Key point 3:</strong> Essential understanding of element C.",
        ]

    def test_summary_combination(self):
        """Test combining summary chunks"""
        result = combine_chunk_results_with_llm(
            results=self.summary_chunks,
            context=self.context,
            combine_prompt=SUMMARY_COMBINE_PROMPT,
            is_summary=True,
        )

        # Basic validation
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Content validation
        self.assertIn("point", result.lower())  # Should contain some mention of points

    def test_blocknotes_combination(self):
        """Test combining blocknote chunks"""
        result = combine_chunk_results_with_llm(
            results=self.blocknote_chunks,
            context=self.context,
            combine_prompt=BLOCKNOTES_COMBINE_PROMPT,
            is_summary=False,
        )

        # Basic validation
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Structure validation
        self.assertTrue(any(tag in result.lower() for tag in ["<strong>", "<em>"]))
        self.assertLessEqual(len(result.split()), 150)  # Should be max 150 words

    def test_single_chunk_handling(self):
        """Test handling of single chunk input"""
        single_chunk = ["This is a single chunk of content."]

        # Test with summary
        summary_result = combine_chunk_results_with_llm(
            results=single_chunk,
            context=self.context,
            combine_prompt=SUMMARY_COMBINE_PROMPT,
            is_summary=True,
        )
        self.assertEqual(summary_result, single_chunk[0])

        # Test with blocknotes
        blocknote_result = combine_chunk_results_with_llm(
            results=single_chunk,
            context=self.context,
            combine_prompt=BLOCKNOTES_COMBINE_PROMPT,
            is_summary=False,
        )
        self.assertEqual(blocknote_result, single_chunk[0])


if __name__ == "__main__":
    unittest.main()
