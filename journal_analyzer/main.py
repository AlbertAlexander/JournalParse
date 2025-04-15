import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import argparse
import os

from .journal_parser import split_journal_entries
from .emotion_analyzer import LLMEmotionAnalyzer
from .database_manager import (
    create_tables, 
    store_emotion_analysis,
    get_db_connection,
    insert_entry
)
from .quantitative_analyzer import calculate_metrics
from .temporal_analyzer import analyze_time_period, analyze_full_journal, store_temporal_analysis
from .config import JOURNAL_INPUT_FILE, DATA_DIR, DEFAULT_LLM_BACKEND, CURRENT_LLM_BACKEND, DEFAULT_LLM_MODEL
from .pronoun_analyzer import analyze_pronouns

# Set logging configuration once at the module level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_database():
    """Initialize the database."""
    logging.info("Setting up database...")
    create_tables()  # This should now show our detailed logging

def process_single_entry(date: datetime.date, content: str, backend: str = "lambda") -> Optional[int]:
    """Process a single journal entry with all analysis methods."""
    try:
        # Standard library analysis - get both metrics and pronoun_metrics
        metrics, pronoun_metrics = calculate_metrics(content)
        
        # Store basic entry and get entry_id using pronoun_metrics from calculate_metrics
        entry_id = insert_entry(date, content, metrics, pronoun_metrics)
        
        if not entry_id:
            raise ValueError("Failed to insert entry")
            
        # LLM emotion analysis - only pass text content
        emotion_analyzer = LLMEmotionAnalyzer()
        emotion_results = emotion_analyzer.analyze_emotion(text=content)
        store_emotion_analysis(entry_id, emotion_results)
        
        return entry_id
        
    except Exception as e:
        logging.error(f"Error processing entry from {date}: {e}")
        return None

def batch_process_entries(journal_path: str = None):
    """Process all entries in journal file."""
    if not journal_path:
        logging.error("No journal path provided")
        return
        
    try:
        # Parse entries from file
        entries = split_journal_entries(journal_path)
        if not entries:
            logging.error("No entries found in journal")
            return
            
        # Initialize counters
        total = len(entries)  # Define total here
        processed = 0
        failed = 0
        
        logging.info(f"Found {total} entries to process")
        
        # Process each entry
        for date, content in entries:
            try:
                entry_id = process_single_entry(date, content)
                
                # Update counters
                if entry_id:
                    processed += 1
                    # Log progress periodically
                    if processed % 10 == 0:
                        logging.info(f"Processed {processed}/{total} entries")
                else:
                    failed += 1
                    logging.warning(f"Entry for {date} was not processed (no entry_id returned)")
                    
            except Exception as e:
                failed += 1
                logging.error(f"Failed to process entry {date}: {e}")
        
        # Final status
        logging.info(f"Processing complete. Successful: {processed}, Failed: {failed}")
        return processed, failed
        
    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        return 0, 0

def run_temporal_analysis(query: str):
    """
    Run analysis across time periods and full journal for a single query.
    
    Args:
        query: The question to analyze
    """
    logging.info(f"Running temporal analysis for query: {query}")
    
    # Analyze by different time periods
    periods = ['year', 'quarter', 'month']
    for period in periods:
        logging.info(f"Running {period} analysis...")
        results = analyze_time_period(period, query)
        if results:
            # Store results in llm_analysis_results table
            store_temporal_analysis(period, results)
            logging.info(f"Completed {period} analysis")
        else:
            logging.error(f"Failed to get results for {period} analysis")
    
    # Full journal analysis
    logging.info("Running full journal analysis...")
    full_results = analyze_full_journal(query)
    if full_results:
        store_temporal_analysis('full', full_results)
        logging.info("Completed full journal analysis")
    else:
        logging.error("Failed to get results for full journal analysis")

def main():
    parser = argparse.ArgumentParser(description='Journal Analysis System')
    parser.add_argument('--setup', action='store_true', help='Initialize database')
    parser.add_argument('--process', action='store_true', help='Process journal entries')
    parser.add_argument('--analyze', action='store_true', help='Run temporal analysis')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--backend', choices=['lambda', 'ollama'], 
                       default=DEFAULT_LLM_BACKEND,
                       help=f'LLM backend to use (default: {DEFAULT_LLM_BACKEND})')
    parser.add_argument('--model', help='Override default model name')
    
    args = parser.parse_args()
    
    # Set global backend configuration
    global CURRENT_LLM_BACKEND
    CURRENT_LLM_BACKEND = args.backend
    
    if args.model:
        os.environ['LAMBDA_MODEL'] = args.model
    
    if args.setup or args.all:
        setup_database()
    
    if args.process or args.all:
        batch_process_entries()  # No backend parameter needed
    
    if args.analyze or args.all:
        # Prompt for query if analyze flag is set
        query = "Analyze emotional patterns and themes in this journal"
        run_temporal_analysis(query)

if __name__ == "__main__":
    main()
