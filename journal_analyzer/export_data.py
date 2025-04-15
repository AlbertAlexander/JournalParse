import csv
import sqlite3
import logging
from pathlib import Path
from .database_manager import get_db_connection

def export_emotion_data(output_file='emotion_data.csv'):
    """Export emotional metrics to CSV file."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get all emotional data with dates
        query = """
        SELECT 
            entry_date,
            valence_score,
            arousal_score,
            content
        FROM entries
        WHERE valence_score IS NOT NULL 
          AND arousal_score IS NOT NULL
        ORDER BY entry_date
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            logging.warning("No emotion data found to export")
            return
            
        # Write to CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['Date', 'Valence', 'Arousal', 'Entry Text'])
            # Write data
            writer.writerows(rows)
            
        logging.info(f"Exported {len(rows)} entries to {output_file}")
        
        # Print summary
        print(f"\nExported {len(rows)} entries to {output_file}")
        print(f"Date range: {rows[0][0]} to {rows[-1][0]}")
        
    except Exception as e:
        logging.error(f"Error exporting data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    export_emotion_data() 