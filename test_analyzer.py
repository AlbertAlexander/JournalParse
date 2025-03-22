import warnings
warnings.filterwarnings('ignore')

from analyzer import JournalAnalyzer
from datetime import datetime
from pathlib import Path
from test_entry import TEST_ENTRY
import json

def test_analyzers():
    """Test all analyzers with sample entry"""
    
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"test_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    
    try:
        analyzer = JournalAnalyzer()
        
        # Get responses for both analysis types
        imagery_response = analyzer.analyze_entry(TEST_ENTRY, "imagery")
        emotions_response = analyzer.analyze_entry(TEST_ENTRY, "emotions")
        
        # Create entry with full response data
        entry_data = {
            "entry_id": "TEST_001",
            "date": datetime.now().isoformat(),
            "content": TEST_ENTRY,
            "analysis_results": {
                "imagery": {
                    "status": imagery_response["status"],
                    "parsed_data": imagery_response["parsed_data"],
                    "raw_response": imagery_response["raw_response"]
                },
                "emotions": {
                    "status": emotions_response["status"],
                    "parsed_data": emotions_response["parsed_data"],
                    "raw_response": emotions_response["raw_response"]
                }
            }
        }
        
        # Save to JSONL file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entry_data, f)
            f.write('\n')
            
        # Print summary
        print(f"\nTest results saved to: {output_file}")
        print("\nAnalysis Summary:")
        print(f"Imagery analysis status: {imagery_response['status']}")
        print(f"Emotions analysis status: {emotions_response['status']}")
        
        if imagery_response['parsed_data']:
            print("\nFound metaphors:")
            for metaphor in imagery_response['parsed_data'].get('metaphors', []):
                print(f"- {metaphor.get('core_metaphor', 'Unknown metaphor')}")
        
        return entry_data
            
    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    test_analyzers()
