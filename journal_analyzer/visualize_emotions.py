import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import sqlite3
import logging
from .database_manager import get_db_connection

def plot_emotional_trends():
    """Create time series visualization of valence and arousal scores."""
    conn = get_db_connection()
    
    try:
        # Get data from database
        query = """
        SELECT 
            entry_date,
            valence_score,
            arousal_score
        FROM entries
        WHERE valence_score IS NOT NULL 
          AND arousal_score IS NOT NULL
        ORDER BY entry_date
        """
        
        # Load into pandas for easier manipulation
        df = pd.read_sql_query(query, conn, 
                             parse_dates=['entry_date'])
        
        if df.empty:
            logging.warning("No emotion data found in database")
            return
            
        # Create the visualization
        plt.figure(figsize=(15, 8))
        
        # Plot both metrics
        plt.plot(df['entry_date'], df['valence_score'], 
                label='Valence', color='blue', alpha=0.7)
        plt.plot(df['entry_date'], df['arousal_score'], 
                label='Arousal', color='red', alpha=0.7)
        
        # Add rolling averages
        window = 7  # 7-day rolling average
        plt.plot(df['entry_date'], 
                df['valence_score'].rolling(window=window).mean(),
                label=f'{window}-day Valence Trend', 
                color='blue', linestyle='--')
        plt.plot(df['entry_date'], 
                df['arousal_score'].rolling(window=window).mean(),
                label=f'{window}-day Arousal Trend', 
                color='red', linestyle='--')
        
        # Customize the plot
        plt.title('Emotional Trends Over Time', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Score (0-10)', fontsize=12)
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
        
        # Add grid and legend
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        
        # Set y-axis limits to match score range
        plt.ylim(0, 10)
        
        # Add horizontal line at neutral (5.0)
        plt.axhline(y=5.0, color='gray', linestyle=':', alpha=0.5)
        
        # Save the plot
        plt.savefig('emotional_trends.png', dpi=300, bbox_inches='tight')
        logging.info("Emotional trends visualization saved as 'emotional_trends.png'")
        
        # Show some statistics
        print("\nEmotional Trends Summary:")
        print(f"Date Range: {df['entry_date'].min().date()} to {df['entry_date'].max().date()}")
        print(f"Number of entries: {len(df)}")
        print("\nValence (positivity):")
        print(f"Average: {df['valence_score'].mean():.2f}")
        print(f"Range: {df['valence_score'].min():.2f} to {df['valence_score'].max():.2f}")
        print("\nArousal (intensity):")
        print(f"Average: {df['arousal_score'].mean():.2f}")
        print(f"Range: {df['arousal_score'].min():.2f} to {df['arousal_score'].max():.2f}")
        
    except Exception as e:
        logging.error(f"Error creating visualization: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    plot_emotional_trends() 