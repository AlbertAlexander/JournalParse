from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta, date
import json

from .llm_manager import query_llm, parse_llm_json_response
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
        
        columns = [description[0] for description in cursor.description]
        entries = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        if not entries:
            logging.warning(f"No entries found for period {start_date} to {end_date}")
            return None
            
        return entries
    finally:
        conn.close()

def analyze_time_period(period_type: str, query_or_queries, start_date: date = None, end_date: date = None):
    """
    Analyze a specific time period with a single query or list of queries.
    
    Args:
        period_type: Type of period ('year', 'quarter', 'month', etc.)
        query_or_queries: Single query string or list of query strings
        start_date: Start date for analysis (optional)
        end_date: End date for analysis (optional)
        
    Returns:
        Dictionary of results, with query strings as keys
    """
    try:
        # Get period boundaries if not specified
        if not (start_date and end_date):
            cursor = get_db_connection().cursor()
            cursor.execute("SELECT MIN(entry_date) as min_date, MAX(entry_date) as max_date FROM entries")
            columns = [description[0] for description in cursor.description]
            date_range = dict(zip(columns, cursor.fetchone()))
            start_date, end_date = date_range['min_date'], date_range['max_date']
        
        # Get entries for period
        entries = get_entries_for_period(start_date, end_date)
        if not entries:
            logging.warning(f"No entries found for period {start_date} to {end_date}")
            return None
            
        # Construct context for LLM
        context = f"""
        Analyzing journal entries from {start_date} to {end_date}.
        Number of entries: {len(entries)}
        
        Key statistics:
        - Average valence: {sum(e['valence_score'] for e in entries if e.get('valence_score') is not None) / len(entries):.2f}
        - Average arousal: {sum(e['arousal_score'] for e in entries if e.get('arousal_score') is not None) / len(entries):.2f}
        """
        
        # Handle both single query and list of queries
        queries = query_or_queries if isinstance(query_or_queries, list) else [query_or_queries]
        results = {}
        
        for query in queries:
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
            
            try:
                response = query_llm(prompt)
                if response:
                    # Add defensive JSON parsing
                    try:
                        results[query] = json.loads(response)
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON parsing error for query '{query[:50]}...': {e}")
                        results[query] = {"error": f"JSON parsing error: {e}", "raw_response": response}
            except Exception as e:
                logging.error(f"Error processing query '{query[:50]}...': {e}")
                results[query] = {"error": str(e)}
        
        # If it was a single query, return just that result
        if not isinstance(query_or_queries, list):
            return results.get(query_or_queries)
            
        return results
        
    except Exception as e:
        log_error(
            analysis_type=f'temporal_{period_type}',
            error=e,
            period_start=start_date,
            period_end=end_date,
            context={'period_type': period_type}
        )
        return None

def analyze_full_journal(query: str) -> Dict:
    """
    Analyze patterns across the entire journal for a single query.
    
    Args:
        query: The question to analyze
        
    Returns:
        Analysis results as a dictionary
    """
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
        columns = [description[0] for description in cursor.description]
        stats = dict(zip(columns, cursor.fetchone()))
        
        if not stats or stats['entry_count'] == 0:
            logging.warning("No entries found for full journal analysis")
            return None
            
        context = f"""
        Analyzing complete journal from {stats['start_date']} to {stats['end_date']}.
        Total entries: {stats['entry_count']}
        Overall emotional trends:
        - Average valence: {stats['avg_valence']:.2f}
        - Average arousal: {stats['avg_arousal']:.2f}
        """
        
        prompt = f"""
        {context}
        
        Question: {query}
        
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
        
        try:
            response = query_llm(prompt)
            if response:
                try:
                    return json.loads(response)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON parsing error in full journal analysis: {e}")
                    return {"error": f"JSON parsing error: {e}", "raw_response": response}
            return None
        except Exception as e:
            logging.error(f"Error in full journal analysis: {e}")
            return {"error": str(e)}
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
    """Analyze journal entries with context from previous analyses."""
    logging.info("Starting temporal analysis")  # Add source logging
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # # Check for exact match
        # cursor.execute("""
        #     SELECT llm_response 
        #     FROM llm_analysis_results 
        #     WHERE question_ref = ?
        #     AND (time_period_start = ? OR ? IS NULL)
        #     AND (time_period_end = ? OR ? IS NULL)
        # """, (query, start_date, start_date, end_date, end_date))
        
        # existing = cursor.fetchone()
        # logging.debug(f"Existing response found: {existing}")
        # if existing:
        #     return json.loads(existing['llm_response'])
        #     logging.info(f"Exact match found for query: {query}")
            
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
                    columns = [description[0] for description in cursor.description]
                    result_dict = dict(zip(columns, result))
                    previous_analyses[ref] = json.loads(result_dict['llm_response'])

        # Build prompt with context
        prompt = construct_prompt(query, entries, previous_analyses)
        logging.debug(f"Found {len(entries) if entries else 0} entries")  # Debug entries
        
        # Get LLM response
        response = query_llm(prompt)
        logging.debug(f"LLM raw response: {response}")
        if not response:
            logging.error("Failed to get LLM response")
            return None
        logging.info(f"LLM response: {response}")
            
        # Store the raw response
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
            response,  # Store raw response
            "default"
        ))
        
        conn.commit()
        
        # Try to parse as JSON if it's in JSON format, otherwise return raw
        required_fields = ['emotional_trajectory', 'recurring_patterns', 'growth_areas', 'key_triggers', 'recommendations']
        defaults = {
            'emotional_trajectory': 'No clear trajectory detected',
            'recurring_patterns': [],
            'growth_areas': [],
            'key_triggers': [],
            'recommendations': []
        }
        
        result = parse_llm_json_response(response, required_fields, defaults)
        
        return result

    except Exception as e:
        logging.error(f"Error in temporal_analyzer: {e}")  # Add source to error
        conn.rollback()
        return None
    finally:
        conn.close()

def construct_prompt(query: str, entries: List[Dict], previous_analyses: Dict = None) -> str:
    """Construct a prompt with context from entries and previous analyses."""
    # Check if entries exist
    if not entries:
        return f"""
        Analyzing journal entries based on the following question:
        {query}

        No entries found for this time period.
        """
    
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
    """
    
    return prompt
