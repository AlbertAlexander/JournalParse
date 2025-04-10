import logging
import json
import traceback
from datetime import datetime, date
from typing import Optional, List, Dict, Union
from .database_manager import get_db_connection

def log_error(
    analysis_type: str,
    error: Exception,
    entry_id: Optional[int] = None,
    period_start: Optional[date] = None,
    period_end: Optional[date] = None,
    context: Dict = None
) -> int:
    """Log an analysis error with full context."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    error_details = {
        'error_type': type(error).__name__,
        'traceback': traceback.format_exc(),
        'context': context or {}
    }
    
    try:
        cursor.execute("""
            INSERT INTO analysis_errors
            (analysis_type, entry_id, period_start, period_end, 
             error_message, error_details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            analysis_type,
            entry_id,
            period_start,
            period_end,
            str(error),
            json.dumps(error_details)
        ))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def get_failed_analyses(
    analysis_type: str,
    include_resolved: bool = False
) -> List[Dict]:
    """Get all failed analyses of a specific type."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT *
            FROM analysis_errors
            WHERE analysis_type = ?
        """
        if not include_resolved:
            query += " AND resolved = FALSE"
        query += " ORDER BY error_timestamp DESC"
        
        cursor.execute(query, (analysis_type,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def mark_resolved(
    error_ids: Union[int, List[int]],
    resolution_notes: str = None
):
    """Mark error(s) as resolved."""
    if isinstance(error_ids, int):
        error_ids = [error_ids]
        
    conn = get_db_connection()
    try:
        conn.executemany("""
            UPDATE analysis_errors
            SET resolved = TRUE,
                resolution_timestamp = CURRENT_TIMESTAMP,
                resolution_notes = ?
            WHERE error_id = ?
        """, [(resolution_notes, error_id) for error_id in error_ids])
        conn.commit()
    finally:
        conn.close()

def get_error_summary() -> Dict:
    """Get summary of unresolved errors by type."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                analysis_type,
                COUNT(*) as error_count,
                COUNT(DISTINCT entry_id) as entries_affected,
                COUNT(DISTINCT period_start) as periods_affected,
                MIN(error_timestamp) as earliest_error,
                MAX(error_timestamp) as latest_error
            FROM analysis_errors
            WHERE resolved = FALSE
            GROUP BY analysis_type
        """)
        return {row['analysis_type']: dict(row) for row in cursor.fetchall()}
    finally:
        conn.close() 