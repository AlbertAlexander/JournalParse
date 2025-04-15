import sqlite3
from pathlib import Path
from datetime import date, datetime
from typing import List, Tuple, Dict, Optional, Any
import logging
import json

from .config import DB_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection() -> sqlite3.Connection:
    """Establishes and returns a database connection."""
    try:
        conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        conn.execute("PRAGMA foreign_keys = ON;") # Enforce foreign key constraints
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise

def create_tables():
    """Creates the database tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Log start of table creation
        logging.info("Starting database table creation...")
        
        # Track tables to create
        tables_to_create = [
            ("entries", """
            CREATE TABLE IF NOT EXISTS entries (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_date DATE NOT NULL,
                content TEXT NOT NULL,
                word_count INTEGER,
                sentence_count INTEGER,
                avg_sentence_length REAL,
                reading_level_flesch REAL,
                sentiment_score_vader REAL,
                sentiment_label_vader TEXT,
                year INTEGER NOT NULL,
                quarter INTEGER NOT NULL,
                month INTEGER NOT NULL,
                week_of_year INTEGER NOT NULL,
                day_of_week INTEGER NOT NULL,
                valence_score REAL,
                arousal_score REAL
            );"""),
            
            ("pronoun_usage", """
            CREATE TABLE IF NOT EXISTS pronoun_usage (
                usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER NOT NULL,
                pronoun_category TEXT NOT NULL,
                count INTEGER NOT NULL,
                percentage REAL NOT NULL,
                FOREIGN KEY (entry_id) REFERENCES entries (entry_id) ON DELETE CASCADE
            );"""),
            
            ("entities", """
            CREATE TABLE IF NOT EXISTS entities (
                entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL
            );"""),
            
            ("entry_entities", """
            CREATE TABLE IF NOT EXISTS entry_entities (
                entry_entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER NOT NULL,
                entity_id INTEGER NOT NULL,
                context_snippet TEXT,
                FOREIGN KEY (entry_id) REFERENCES entries (entry_id) ON DELETE CASCADE,
                FOREIGN KEY (entity_id) REFERENCES entities (entity_id) ON DELETE CASCADE,
                UNIQUE(entry_id, entity_id)
            );"""),
            
            ("emotion_analysis", """
            CREATE TABLE IF NOT EXISTS emotion_analysis (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id INTEGER NOT NULL,
                primary_emotions TEXT NOT NULL,  -- JSON array of emotions
                emotional_patterns TEXT,
                analysis_confidence REAL,
                llm_reasoning TEXT,
                FOREIGN KEY (entry_id) REFERENCES entries (entry_id) ON DELETE CASCADE
            );"""),
            
            ("llm_analysis_results", """
            CREATE TABLE IF NOT EXISTS llm_analysis_results (
                analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_ref TEXT,
                time_period_start DATE,
                time_period_end DATE,
                prompt_summary TEXT,
                llm_response TEXT NOT NULL,
                model_used TEXT,
                analysis_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );""")
        ]
        
        # Create each table and log the process
        for table_name, create_sql in tables_to_create:
            logging.info(f"Creating table: {table_name}")
            try:
                cursor.execute(create_sql)
                logging.info(f"Successfully created table: {table_name}")
            except sqlite3.Error as e:
                logging.error(f"Error creating {table_name} table: {e}")
                raise
        
        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [t[0] for t in cursor.fetchall()]
        logging.info(f"Tables in database after creation: {existing_tables}")
        
        # Check for missing tables
        expected_tables = {t[0] for t in tables_to_create}
        missing_tables = expected_tables - set(existing_tables)
        if missing_tables:
            logging.error(f"Failed to create tables: {missing_tables}")
            raise Exception(f"Missing tables after creation: {missing_tables}")
            
        conn.commit()
        logging.info("All database tables created and committed successfully")
        
    except sqlite3.Error as e:
        logging.error(f"SQLite error during table creation: {str(e)}")
        conn.rollback()
        raise
    except Exception as e:
        logging.error(f"Unexpected error during table creation: {str(e)}")
        conn.rollback()
        raise
    finally:
        conn.close()

def insert_entry(entry_date: date, content: str, metrics: Dict[str, Any], pronoun_data: Dict[str, Dict[str, Any]]):
    """Inserts a single journal entry and its associated metrics and pronoun usage."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Calculate time period fields
        year = entry_date.year
        month = entry_date.month
        day_of_week = entry_date.weekday() # Monday is 0, Sunday is 6
        week_of_year = entry_date.isocalendar()[1]
        quarter = (month - 1) // 3 + 1

        cursor.execute("""
        INSERT INTO entries (
            entry_date, content, word_count, sentence_count, avg_sentence_length,
            reading_level_flesch, sentiment_score_vader, sentiment_label_vader,
            year, quarter, month, week_of_year, day_of_week
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry_date, content, metrics.get('word_count'), metrics.get('sentence_count'),
            metrics.get('avg_sentence_length'), metrics.get('reading_level_flesch'),
            metrics.get('sentiment_score_vader'), metrics.get('sentiment_label_vader'),
            year, quarter, month, week_of_year, day_of_week
        ))
        entry_id = cursor.lastrowid

        # Insert pronoun usage
        if entry_id and pronoun_data:
            pronoun_rows = []
            for category, data in pronoun_data.items():
                pronoun_rows.append((entry_id, category, data['count'], data['percentage']))
            cursor.executemany("""
            INSERT INTO pronoun_usage (entry_id, pronoun_category, count, percentage)
            VALUES (?, ?, ?, ?)
            """, pronoun_rows)

        conn.commit()
        logging.debug(f"Inserted entry ID: {entry_id} for date {entry_date}")
        return entry_id
    except sqlite3.Error as e:
        logging.error(f"Error inserting entry for date {entry_date}: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_entries_by_date_range(start_date: date, end_date: date) -> List[sqlite3.Row]:
    """Retrieves entries within a specified date range."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT entry_id, entry_date, content, word_count
        FROM entries
        WHERE entry_date BETWEEN ? AND ?
        ORDER BY entry_date ASC
        """, (start_date, end_date))
        entries = cursor.fetchall()
        logging.info(f"Retrieved {len(entries)} entries between {start_date} and {end_date}")
        return entries
    except sqlite3.Error as e:
        logging.error(f"Error retrieving entries for range {start_date} - {end_date}: {e}")
        return []
    finally:
        conn.close()

def insert_llm_analysis(question_ref: str, start_date: date, end_date: date,
                        prompt_summary: str, llm_response: str, model_used: str):
    """Stores the result of an LLM analysis query."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO llm_analysis_results (
            question_ref, time_period_start, time_period_end, prompt_summary,
            llm_response, model_used
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (question_ref, start_date, end_date, prompt_summary, llm_response, model_used))
        conn.commit()
        analysis_id = cursor.lastrowid
        logging.info(f"Stored LLM analysis result ID: {analysis_id}")
        return analysis_id
    except sqlite3.Error as e:
        logging.error(f"Error storing LLM analysis result: {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def get_quantitative_trend(metric: str, period: str, start_date: date, end_date: date) -> List[Tuple[str, float]]:
    """
    Retrieves aggregated quantitative metrics over time periods.

    Args:
        metric: The column name of the metric in the 'entries' table (e.g., 'sentiment_score_vader').
        period: The time period to group by ('year', 'quarter', 'month', 'week').
        start_date: The start date of the analysis range.
        end_date: The end date of the analysis range.

    Returns:
        A list of tuples, where each tuple is (period_label, average_metric_value).
        Example: [('2014-01', -0.25), ('2014-02', 0.15), ...]
    """
    if period not in ['year', 'month', 'quarter', 'week']:
        raise ValueError("Invalid period. Choose 'year', 'quarter', 'month', or 'week'.")
    if metric not in ['word_count', 'sentence_count', 'avg_sentence_length', 'reading_level_flesch', 'sentiment_score_vader']:
         raise ValueError(f"Invalid metric column: {metric}")


    conn = get_db_connection()
    cursor = conn.cursor()

    # Construct the grouping and formatting based on the period
    if period == 'year':
        group_by_clause = "strftime('%Y', entry_date)"
        select_clause = "strftime('%Y', entry_date) as period_label"
    elif period == 'month':
        group_by_clause = "strftime('%Y-%m', entry_date)"
        select_clause = "strftime('%Y-%m', entry_date) as period_label"
    elif period == 'quarter':
        # SQLite doesn't have a direct quarter function, group by year and quarter number
        group_by_clause = "year, quarter"
        select_clause = "printf('%d-Q%d', year, quarter) as period_label"
    elif period == 'week':
        # Use ISO week date format YYYY-Www
        group_by_clause = "strftime('%Y-W%W', entry_date)"
        select_clause = "strftime('%Y-W%W', entry_date) as period_label"

    query = f"""
    SELECT
        {select_clause},
        AVG({metric}) as avg_value
    FROM entries
    WHERE entry_date BETWEEN ? AND ?
    GROUP BY {group_by_clause}
    ORDER BY period_label ASC;
    """

    try:
        cursor.execute(query, (start_date, end_date))
        results = cursor.fetchall()
        # Convert results from sqlite3.Row to simple tuples
        trend_data = [(row['period_label'], row['avg_value']) for row in results]
        logging.info(f"Retrieved {len(trend_data)} data points for {metric} trend by {period}.")
        return trend_data
    except sqlite3.Error as e:
        logging.error(f"Error retrieving trend data for {metric} by {period}: {e}")
        return []
    finally:
        conn.close()

# --- Entity Functions ---
# Note: Entity management functions have been moved to entity_manager.py
# Use the functions from entity_manager.py instead

def store_emotion_analysis(entry_id: int, analysis: Dict):
    """Store emotion analysis results."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Update basic scores in entries table
        cursor.execute("""
            UPDATE entries 
            SET valence_score = ?,
                arousal_score = ?
            WHERE entry_id = ?
        """, (analysis['valence'], analysis['arousal'], entry_id))
        
        # Store detailed analysis
        cursor.execute("""
            INSERT INTO emotion_analysis (
                entry_id,
                primary_emotions,
                emotional_patterns,
                analysis_confidence,
                llm_reasoning
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            entry_id,
            json.dumps(analysis['primary_emotions']),
            analysis['emotional_patterns'],
            analysis['confidence'],
            analysis['reasoning']
        ))
        
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error storing emotion analysis: {e}")
        conn.rollback()
    finally:
        conn.close() 