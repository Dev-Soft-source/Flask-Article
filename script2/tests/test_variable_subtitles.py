import sys
import os
from pathlib import Path

# Add the utils directory to Python path
script_dir = Path(__file__).parent.parent
sys.path.append(str(script_dir))

from utils.unified_csv_processor import UnifiedCSVProcessor

def test_variable_subtitles():
    # Initialize the CSV processor with test file
    csv_path = Path(__file__).parent / "test_variable_subtitles.csv"
    processor = UnifiedCSVProcessor(str(csv_path))
    
    # Process the file
    data = processor.process_file()
    
    # Print the structure of each article
    for index, article_data in data.items():
        print(f"\nArticle {index}:")
        print(f"Keyword: {article_data.get('keyword')}")
        print(f"Featured Image: {article_data.get('featured_img')}")
        
        # Get subtitle-image pairs
        pairs = []
        i = 1
        while True:
            subtitle_key = f'Subtitle{i}'
            img_key = f'img{i}'
            
            if subtitle_key not in article_data or not article_data[subtitle_key].strip():
                break
                
            pairs.append({
                'subtitle': article_data[subtitle_key],
                'image': article_data[img_key]
            })
            i += 1
            
        print(f"Number of subtitle-image pairs: {len(pairs)}")
        for j, pair in enumerate(pairs, 1):
            print(f"  {j}. Subtitle: {pair['subtitle']}")
            print(f"     Image: {pair['image']}")

if __name__ == "__main__":
    test_variable_subtitles()
