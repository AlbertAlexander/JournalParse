import argparse
from src.database import JournalDB

def main():
    parser = argparse.ArgumentParser(description='Query journal database')
    parser.add_argument('--name', help='Find entries with name reference')
    parser.add_argument('--sentiment', choices=['high', 'low'], help='Find entries with high/low sentiment')
    parser.add_argument('--date-range', help='Date range (YYYY-MM-DD:YYYY-MM-DD)')
    
    args = parser.parse_args()
    db = JournalDB()
    
    if args.name:
        entries = db.get_entries_with_reference(args.name)
        print(f"Found {len(entries)} entries mentioning {args.name}")
        for entry in entries:
            print(f"{entry['date']}: {entry['raw_text'][:100]}...")
    
    # Add more query options
    
if __name__ == "__main__":
    main()
