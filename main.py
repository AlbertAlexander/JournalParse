import os
import sys
from pathlib import Path
import json
from typing import Dict, Optional
from enum import Enum
import argparse

# Add the project directory to the Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from models import JournalChunk, JournalCollection, ChunkSize
from split import split_journal_into_chunks, process_test_sample
from sentiment_analyzer import SentimentAnalyzer
from analyzer import JournalAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class OperationMode(Enum):
    INSERT = "insert"  # Only add new entries
    UPDATE = "update"  # Update existing entries with new fields
    OVERWRITE = "overwrite"  # Replace existing entries completely

class JournalProcessor:
    def __init__(self, output_dir: str = "analysis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def process_entry(self, entry_data: Dict, operation: OperationMode, analysis_type: str) -> None:
        """Process a single entry based on operation mode"""
        file_path = self.output_dir / f"journal_analysis.jsonl"
        
        # Load existing entries if needed
        existing_entries = {}
        if file_path.exists() and operation != OperationMode.OVERWRITE:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    existing_entries[entry['entry_id']] = entry
        
        # Handle entry based on operation mode
        if operation == OperationMode.INSERT:
            if entry_data['entry_id'] not in existing_entries:
                self._append_entry(entry_data, file_path)
                
        elif operation == OperationMode.UPDATE:
            if entry_data['entry_id'] in existing_entries:
                # Update only the new analysis type
                existing_entries[entry_data['entry_id']][analysis_type] = entry_data[analysis_type]
                self._rewrite_entries(existing_entries, file_path)
            else:
                self._append_entry(entry_data, file_path)
                
        else:  # OVERWRITE
            self._append_entry(entry_data, file_path)
    
    def _append_entry(self, entry_data: Dict, file_path: Path) -> None:
        """Append a single entry to JSONL file"""
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(entry_data, f)
            f.write('\n')
    
    def _rewrite_entries(self, entries: Dict[str, Dict], file_path: Path) -> None:
        """Rewrite entire JSONL file with updated entries"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in entries.values():
                json.dump(entry, f)
                f.write('\n')

def test_processing():
    processor = JournalProcessor()
    
    # First pass - VADER analysis
    for chunk in journal.chunks:
        entry_data = {
            "entry_id": chunk.chunk_id,
            "date": chunk.date.isoformat(),
            "content": chunk.content,
            "vader_analysis": vader_analyzer.polarity_scores(chunk.content)
        }
        processor.process_entry(entry_data, OperationMode.INSERT, "vader_analysis")
    
    # Second pass - Sentiment analysis
    for chunk in journal.chunks:
        entry_data = {
            "entry_id": chunk.chunk_id,
            "sentiment_analysis": sentiment_analyzer.analyze_entry_with_subchunks(chunk)
        }
        processor.process_entry(entry_data, OperationMode.UPDATE, "sentiment_analysis")
    
    # Third pass - Claude analysis
    for chunk in journal.chunks:
        entry_data = {
            "entry_id": chunk.chunk_id,
            "claude_analysis": journal_analyzer.analyze_entry(chunk.content)
        }
        processor.process_entry(entry_data, OperationMode.UPDATE, "claude_analysis")

def process_journal(file_path: str) -> JournalCollection:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    journal = split_journal_into_chunks(text)
    return journal

def main():
    """Process entire journal with multiple analyzers"""
    parser = argparse.ArgumentParser(description='Process journal entries')
    parser.add_argument('--file', '-f', type=str, required=True,
                       help='Path to journal file')
    args = parser.parse_args()
    
    # Initialize processor and analyzer
    processor = JournalProcessor()
    analyzer = JournalAnalyzer()
    
    # Load and split journal
    with open(args.file, 'r', encoding='utf-8') as f:
        journal = split_journal_into_chunks(f.read())
    
    print(f"Processing {len(journal.chunks)} entries...")
    
    # Process each chunk
    for chunk in journal.chunks:
        entry_data = {
            "entry_id": chunk.chunk_id,
            "date": chunk.date.isoformat(),
            "content": chunk.content,
            "analysis_results": {
                "imagery": analyzer.analyze_entry(chunk.content, "imagery"),
                "emotions": analyzer.analyze_entry(chunk.content, "emotions")
            }
        }
        processor.process_entry(entry_data, OperationMode.INSERT, "analysis_results")
        print(f"Processed entry {chunk.chunk_id}")

if __name__ == "__main__":
    main()  # Uncomment this line
    # test_processing()  # Comment out or remove this line
