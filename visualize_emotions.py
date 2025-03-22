import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

def load_jsonl(file_path):
    """Load JSONL file into a list of dictionaries"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def process_emotions(entries):
    """Extract emotional dimensions and dates from entries"""
    dates = []
    emotions = {
        'valence': [],
        'arousal': [],
        'control': []
    }
    
    for entry in entries:
        date = datetime.fromisoformat(entry['date'])
        dates.append(date)
        
        emotion_data = entry['analysis_results']['emotions']['parsed_data']['emotional_dimensions']
        for dim in emotions.keys():
            emotions[dim].append(emotion_data[dim])
    
    # Convert to numpy arrays
    for dim in emotions:
        emotions[dim] = np.array(emotions[dim])
    
    return dates, emotions

def calculate_moving_average(data, window=7):
    """Calculate moving average with specified window"""
    return pd.Series(data).rolling(window=window, center=True).mean()

def plot_emotions(jsonl_path):
    """Create visualizations of emotional dimensions with moving averages"""
    # Load and process data
    entries = load_jsonl(jsonl_path)
    dates, emotions = process_emotions(entries)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Raw emotional dimensions with moving averages
    for dim, values in emotions.items():
        # Plot raw data
        ax1.plot(dates, values, 'o', alpha=0.3, label=f'{dim} (raw)')
        
        # Calculate and plot moving average
        ma = calculate_moving_average(values)
        ax1.plot(dates, ma, '-', linewidth=2, label=f'{dim} (7-day MA)')
    
    ax1.set_title('Emotional Dimensions Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Intensity (0-10)')
    ax1.set_ylim(0, 10)  # Force full range
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    
    # Plot 2: Moving averages only (cleaner view)
    for dim, values in emotions.items():
        ma = calculate_moving_average(values)
        ax2.plot(dates, ma, '-', linewidth=2, label=f'{dim}')
    
    ax2.set_title('7-Entry Moving Averages')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Intensity (0-10)')
    ax2.set_ylim(0, 10)  # Force full range
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    
    # Plot 3: Valence-Arousal scatter with Control as color
    scatter = ax3.scatter(emotions['valence'], 
                         emotions['arousal'],
                         c=emotions['control'],
                         cmap='viridis',
                         s=100,
                         alpha=0.6)
    
    ax3.set_title('Valence-Arousal Space (color = Control)')
    ax3.set_xlabel('Valence')
    ax3.set_ylabel('Arousal')
    ax3.set_xlim(0, 10)  # Force full range for x-axis
    ax3.set_ylim(0, 10)  # Force full range for y-axis
    ax3.grid(True)
    
    # Calculate statistics
    stats = {
        'mean': {dim: np.mean(values) for dim, values in emotions.items()},
        'std': {dim: np.std(values) for dim, values in emotions.items()},
        'moving_averages': {
            dim: calculate_moving_average(values).tolist() 
            for dim, values in emotions.items()
        }
    }
    
    # Print summary statistics
    print("\nEmotional Statistics:")
    for dim in emotions:
        print(f"\n{dim.capitalize()}:")
        print(f"  Mean: {stats['mean'][dim]:.2f}")
        print(f"  Std:  {stats['std'][dim]:.2f}")
    
    plt.tight_layout()
    plt.show()
    plt.savefig('emotion_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    return stats

if __name__ == "__main__":
    jsonl_path = "analysis_output/journal_analysis.jsonl"
    stats = plot_emotions(jsonl_path)