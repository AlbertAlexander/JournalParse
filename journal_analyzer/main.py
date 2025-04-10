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
    insert_entry,
    get_failed_analyses,
    get_entry,
    mark_resolved
)
from .quantitative_analyzer import calculate_metrics
from .temporal_analyzer import analyze_time_period, analyze_full_journal, store_temporal_analysis
from .config import JOURNAL_INPUT_FILE, DATA_DIR, DEFAULT_LLM_BACKEND, CURRENT_LLM_BACKEND, DEFAULT_LLM_MODEL

logging.basicConfig(level=logging.INFO)

def setup_database():
    """Initialize database and required tables."""
    create_tables()
    logging.info("Database initialized")

def process_single_entry(date: datetime.date, content: str, backend: str = "lambda") -> Optional[int]:
    """Process a single journal entry with all analysis methods."""
    try:
        # Standard library analysis
        metrics = calculate_metrics(content)
        
        # Store basic entry and get entry_id
        entry_id = insert_entry(date, content, metrics)
        
        if not entry_id:
            raise ValueError("Failed to insert entry")
            
        # LLM emotion analysis
        emotion_analyzer = LLMEmotionAnalyzer(backend=backend)
        emotion_results = emotion_analyzer.analyze_emotion(content)
        store_emotion_analysis(entry_id, emotion_results)
        
        return entry_id
        
    except Exception as e:
        logging.error(f"Error processing entry from {date}: {e}")
        return None

def batch_process_entries(backend: str = "lambda"):
    """Process all entries from the journal file."""
    entries = split_journal_entries(JOURNAL_INPUT_FILE)
    total = len(entries)
    logging.info(f"Found {total} entries to process")
    
    processed = 0
    failed = 0
    
    for date, content in entries:
        try:
            entry_id = process_single_entry(date, content, backend=backend)
            if entry_id:
                processed += 1
            else:
                failed += 1
            
            if processed % 10 == 0:  # Progress update every 10 entries
                logging.info(f"Processed {processed}/{total} entries")
                
        except Exception as e:
            failed += 1
            logging.error(f"Failed to process entry {date}: {e}")
            continue
    
    logging.info(f"Processing complete. Successful: {processed}, Failed: {failed}")
    return processed, failed

def run_temporal_analysis(backend: str):
    """Run analysis across time periods and full journal."""
    analyzer = LLMEmotionAnalyzer()
    
    # Load questions from file
    questions_file = DATA_DIR / "questions.txt"
    with open(questions_file, 'r') as f:
        questions = [q.strip() for q in f.readlines() if q.strip()]
    
    # Analyze by different time periods
    periods = ['year', 'quarter', 'month']
    for period in periods:
        results = analyze_time_period(period, questions)
        # Store results in llm_analysis_results table
        store_temporal_analysis(period, results)
    
    # Full journal analysis
    full_results = analyze_full_journal(questions)
    store_temporal_analysis('full', full_results)

def rerun_failed_analyses(analysis_type: str, model: str = DEFAULT_LLM_MODEL):
    """Rerun all failed analyses of a specific type."""
    emotion_analyzer = LLMEmotionAnalyzer(model=model)
    failed = get_failed_analyses(analysis_type)
    logging.info(f"Found {len(failed)} failed {analysis_type} analyses to rerun")
    
    for error in failed:
        try:
            if error['entry_id']:
                # Entry-based analysis
                entry = get_entry(error['entry_id'])
                if analysis_type == 'entry_emotion':
                    result = emotion_analyzer.analyze_emotion(
                        error['entry_id'], 
                        entry['content']
                    )
                    if result:
                        mark_resolved(error['error_id'], resolution_notes=str(result))
                        logging.info(f"Successfully reran analysis for entry {error['entry_id']}")
            else:
                # Period-based analysis
                if analysis_type.startswith('temporal_'):
                    result = analyze_time_period(
                        analysis_type.split('_')[1],
                        error['period_start'],
                        error['period_end']
                    )
                    
            if result:
                mark_resolved(
                    error['error_id'], 
                    "Successfully reprocessed"
                )
                
        except Exception as e:
            logging.error(f"Rerun failed for error_id {error['error_id']}: {e}")

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
        run_temporal_analysis()  # No backend parameter needed

if __name__ == "__main__":
    main()
