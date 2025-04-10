from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta, date
import json

from .llm_manager import query_llm
from .database_manager import get_db_connection
from .error_manager import log_error

def get_entries_for_period(start_date: datetime, end_date: datetime) -> List[Dict]:
    """Retrieve entries for a specific time period."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT entry_date, content, valence_score, arousal_score
            FROM entries
            WHERE entry_date BETWEEN ? AND ?
            ORDER BY entry_date
        """, (start_date, end_date))
        return cursor.fetchall()
    finally:
        conn.close()

def analyze_time_period(period_type: str, query: str, start_date: date, end_date: date):
    """Analyze a specific time period with a single query."""
    try:
        # Get period boundaries if not specified
        if not (start_date and end_date):
            cursor = get_db_connection().cursor()
            cursor.execute("SELECT MIN(entry_date), MAX(entry_date) FROM entries")
            start_date, end_date = cursor.fetchone()
        
        # Get entries for period
        entries = get_entries_for_period(start_date, end_date)
        
        # Construct context for LLM
        context = f"""
        Analyzing journal entries from {start_date} to {end_date}.
        Number of entries: {len(entries)}
        
        Key statistics:
        - Average valence: {sum(e['valence_score'] for e in entries) / len(entries):.2f}
        - Average arousal: {sum(e['arousal_score'] for e in entries) / len(entries):.2f}
        """
        
        prompt = f"""
        {context}
        
        Question: {query}
        
        Analyze these journal entries and provide a detailed response.
        Consider patterns, trends, and significant changes over this time period.
        
        Respond in JSON format:
        {{
            "analysis": str,  // Your detailed analysis
            "key_findings": [str],  // List of main points
            "evidence": [str],  // Supporting examples
            "confidence": float  // Your confidence in the analysis (0-1)
        }}
        """
        
        response = query_llm(prompt)
        if response:
            return json.loads(response)
            
        return None
        
    except Exception as e:
        log_error(
            analysis_type=f'temporal_{period_type}',
            error=e,
            period_start=start_date,
            period_end=end_date,
            context={'period_type': period_type}
        )
        return None

def analyze_full_journal(questions: List[str]) -> Dict:
    """Analyze patterns across the entire journal."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get overall statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as entry_count,
                AVG(valence_score) as avg_valence,
                AVG(arousal_score) as avg_arousal,
                MIN(entry_date) as start_date,
                MAX(entry_date) as end_date
            FROM entries
        """)
        stats = cursor.fetchone()
        
        context = f"""
        Analyzing complete journal from {stats['start_date']} to {stats['end_date']}.
        Total entries: {stats['entry_count']}
        Overall emotional trends:
        - Average valence: {stats['avg_valence']:.2f}
        - Average arousal: {stats['avg_arousal']:.2f}
        """
        
        results = {}
        for question in questions:
            prompt = f"""
            {context}
            
            Question: {question}
            
            Provide a comprehensive analysis of the entire journal.
            Consider long-term patterns, major themes, and significant transitions.
            
            Respond in JSON format:
            {{
                "analysis": str,
                "key_patterns": [str],
                "significant_periods": [str],
                "overall_trajectory": str,
                "confidence": float
            }}
            """
            
            response = query_llm(prompt)
            if response:
                results[question] = json.loads(response)
            
        return results
        
    finally:
        conn.close()

def store_temporal_analysis(period: str, results: Dict):
    """Store temporal analysis results in database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO llm_analysis_results 
            (question_ref, time_period_start, time_period_end, 
             prompt_summary, llm_response, model_used)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            f"temporal_analysis_{period}",
            results.get('period_start'),
            results.get('period_end'),
            f"Temporal analysis for {period}",
            json.dumps(results),
            "ollama_default"
        ))
        conn.commit()
    except Exception as e:
        logging.error(f"Error storing temporal analysis: {e}")
        conn.rollback()
    finally:
        conn.close()

def analyze_with_context(
    query: str,
    start_date: datetime = None,
    end_date: datetime = None,
    include_analyses: List[str] = None,
    model: str = None
) -> Dict:
    """
    Analyze journal entries with optional context from previous analyses.
    Exact string matching on query for finding/updating existing analyses.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check for exact match
        cursor.execute("""
            SELECT llm_response 
            FROM llm_analysis_results 
            WHERE question_ref = ?
            AND (time_period_start = ? OR ? IS NULL)
            AND (time_period_end = ? OR ? IS NULL)
        """, (query, start_date, start_date, end_date, end_date))
        
        existing = cursor.fetchone()
        if existing:
            return json.loads(existing['llm_response'])
            
        # If no exact match, proceed with new analysis
        entries = get_entries_for_period(start_date, end_date)
        previous_analyses = {}
        if include_analyses:
            for ref in include_analyses:
                cursor.execute("""
                    SELECT question_ref, llm_response 
                    FROM llm_analysis_results 
                    WHERE question_ref LIKE ?
                    AND (time_period_start IS NULL OR time_period_start >= ?)
                    AND (time_period_end IS NULL OR time_period_end <= ?)
                    ORDER BY analysis_timestamp DESC
                    LIMIT 1
                """, (f"%{ref}%", start_date, end_date))
                
                result = cursor.fetchone()
                if result:
                    previous_analyses[ref] = json.loads(result['llm_response'])

        # Build prompt with context
        prompt = construct_prompt(query, entries, previous_analyses)
        
        # Get LLM response
        response = query_llm(prompt, model=model)
        
        # Store new result
        cursor.execute("""
            INSERT INTO llm_analysis_results 
            (question_ref, time_period_start, time_period_end,
             prompt_summary, llm_response, model_used)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            query,
            start_date,
            end_date,
            prompt[:100],
            response,
            model
        ))
        
        conn.commit()
        return json.loads(response)

    finally:
        conn.close()

def construct_prompt(query: str, entries: List[Dict], previous_analyses: Dict = None) -> str:
    """Construct a prompt with context from entries and previous analyses."""
    # Format entries for context
    entries_text = "\n---\n".join(
        f"Date: {e['entry_date']}\n{e['content'][:500]}..." 
        for e in entries
    )
    
    context = f"""
    Analyzing journal entries based on the following question:
    {query}

    Time period: {entries[0]['entry_date']} to {entries[-1]['entry_date']}
    Number of entries: {len(entries)}
    """
    
    # Add previous analyses if available
    if previous_analyses:
        context += "\nRelevant previous analyses:\n"
        for ref, analysis in previous_analyses.items():
            context += f"\n{ref}:\n{json.dumps(analysis, indent=2)}\n"
    
    prompt = f"""
    {context}

    Journal entries:
    {entries_text}

    Provide a comprehensive analysis in JSON format:
    {{
        "analysis": str,  // Detailed analysis addressing the question
        "key_findings": [str],  // List of main points
        "evidence": [str],  // Supporting examples from the entries
        "confidence": float  // Your confidence in the analysis (0-1)
    }}
    """
    
    return prompt
