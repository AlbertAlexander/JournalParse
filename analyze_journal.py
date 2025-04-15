from journal_analyzer.main import setup_database, batch_process_entries
from journal_analyzer.temporal_analyzer import analyze_with_context
from journal_analyzer.database_manager import create_tables, get_db_connection
from journal_analyzer.config import CURRENT_LLM_BACKEND, DEFAULT_LLM_BACKEND, DEFAULT_LLM_MODEL, CLI_SELECTED_MODEL
from pathlib import Path
import argparse
import os
from dotenv import load_dotenv
import logging

def print_db_contents(table_name: str = None):
    """
    Print contents of specified table or list all tables if none specified.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if not table_name:
            # List all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print("\nAvailable tables:")
            for table in tables:
                print(f"- {table[0]}")
            return

        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get data
        cursor.execute(f"SELECT * FROM {table_name};")
        rows = cursor.fetchall()
        
        print(f"\nContents of {table_name}:")
        print("Columns:", columns)
        print("\nRows:")
        for row in rows:
            print(dict(zip(columns, row)))
            
    except Exception as e:
        print(f"Error accessing database: {e}")
    finally:
        conn.close()

def analyze_journal(journal_path: str = None, query: str = None):
    """
    Analyze a journal text file with a single query.
    
    Args:
        journal_path: Path to text file containing journal entries
        query: Single analysis question to process

    Note that model names are different on Lambda api endpoint than in Ollama
    """
    load_dotenv()
    
    # Declare global before any use
    global CLI_SELECTED_MODEL
    
    # Now we can reset it
    CLI_SELECTED_MODEL = None

    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['lambda', 'ollama'], 
                       default=DEFAULT_LLM_BACKEND)
    parser.add_argument('--model', type=str, help='LLM model to use (overrides default)')
    parser.add_argument('--journal', type=str,
                       help='Path to journal text file')
    parser.add_argument('--query', type=str,
                       help='Analysis question to process')
    parser.add_argument('--reload', action='store_true',
                       help='Force reload journal into database')
    parser.add_argument('--show-table', type=str,
                       help='Print contents of specified table')
    parser.add_argument('--list-tables', action='store_true',
                       help='List all available tables')
    args = parser.parse_args()

    if args.list_tables:
        print_db_contents()
        return
        
    if args.show_table:
        print_db_contents(args.show_table)
        return

    # Only require journal and query for analysis
    if not args.journal or not args.query:
        if not (args.list_tables or args.show_table):
            parser.error("--journal and --query are required for analysis")
        return

    # Set global backend configuration
    global CURRENT_LLM_BACKEND
    CURRENT_LLM_BACKEND = args.backend
    
    # Update CLI model selection if specified
    if args.model:
        CLI_SELECTED_MODEL = args.model
        logging.info(f"Using CLI-specified model: {CLI_SELECTED_MODEL}")
    else:
        logging.info(f"Using default model: {DEFAULT_LLM_MODEL}")

    # Setup and load journal if needed
    setup_database()
    if args.reload:
        batch_process_entries(journal_path=args.journal)  # Pass journal path here
    
    # Run single query analysis
    result = analyze_with_context(
        query=args.query,
        start_date=None,  # Will analyze full journal
        end_date=None
    )
    
    return result

if __name__ == "__main__":
    # Example usage
    result = analyze_journal(
        journal_path='data/"Journal first 725 anon.txt"',  # Remove hardcoded path
        query="How does the author express emotions throughout the journal?"  # Remove hardcoded query
    )
    print(result)