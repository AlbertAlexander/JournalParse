import os
from pathlib import Path
from src.database import JournalDB
from src.parser import parse_journal_file
from src.analysis.programmatic import analyze_entry_basic

def main():
    # Initialize database
    db = JournalDB()
    
    # Parse journal file -- TODO: make CLI argument
    entries = parse_journal_file('data/raw/journal.txt')
    print(f"Found {len(entries)} entries")
    
    # Process each entry
    for i, entry in enumerate(entries):
        # Run basic analysis
        analysis = analyze_entry_basic(entry['raw_text'])
        entry['analysis'] = {
            'programmatic': analysis
        }
        
        # Save to database
        db.save_entry(entry)
        
        if i % 100 == 0:
            print(f"Processed {i+1}/{len(entries)} entries")
    
    print("Database build complete")

if __name__ == "__main__":
    main()
