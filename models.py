from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict
import statistics

class ChunkSize(Enum):
    NORMAL = "normal"      # <= 1500 words
    LONG = "long"         # 1501-3000 words
    TOO_LONG = "too_long" # > 3000 words

class Sentiment(Enum):
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

@dataclass
class TextMetrics:
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    avg_word_length: float
    reading_level: float  # Flesch-Kincaid score
    unique_words: int
    vocabulary_richness: float  # ratio of unique words to total words

@dataclass
class SentimentMetrics:
    overall_sentiment: Sentiment
    emotional_intensity: float  # 0-1 scale
    key_emotions: List[str]
    confidence_score: float

@dataclass
class JournalChunk:
    # Core data
    date: datetime
    content: str
    chunk_id: str
    
    # Size tracking
    word_count: int
    size_category: ChunkSize
    
    # Chunk organization
    sub_chunk_index: Optional[int] = None
    total_sub_chunks: Optional[int] = None
    
    # Text analysis
    text_metrics: Optional[TextMetrics] = None
    sentiment_metrics: Optional[SentimentMetrics] = None
    
    # References and entities
    people_mentioned: List[str] = None
    places_mentioned: List[str] = None
    cross_references: List[datetime] = None  # dates referenced
    
    # Topics and themes
    primary_topics: List[str] = None
    themes: List[str] = None
    
    def __post_init__(self):
        if self.people_mentioned is None:
            self.people_mentioned = []
        if self.places_mentioned is None:
            self.places_mentioned = []
        if self.cross_references is None:
            self.cross_references = []
        if self.primary_topics is None:
            self.primary_topics = []
        if self.themes is None:
            self.themes = []

@dataclass
class JournalCollection:
    chunks: List[JournalChunk]
    
    def get_chunk_by_date(self, target_date: datetime) -> Optional[JournalChunk]:
        return next((chunk for chunk in self.chunks if chunk.date == target_date), None)
    
    def get_chunks_in_range(self, start_date: datetime, end_date: datetime) -> List[JournalChunk]:
        return [chunk for chunk in self.chunks if start_date <= chunk.date <= end_date]
    
    def get_average_sentiment(self) -> float:
        sentiments = [chunk.sentiment_metrics.overall_sentiment.value 
                     for chunk in self.chunks 
                     if chunk.sentiment_metrics]
        return statistics.mean(sentiments) if sentiments else 0
    
    def get_reading_level_trend(self) -> Dict[datetime, float]:
        return {chunk.date: chunk.text_metrics.reading_level 
                for chunk in self.chunks 
                if chunk.text_metrics}
    
    def find_related_entries(self, chunk: JournalChunk) -> List[JournalChunk]:
        """Find entries that share topics, people, or are cross-referenced"""
        related = []
        for other_chunk in self.chunks:
            if other_chunk == chunk:
                continue
            
            # Check for shared people
            shared_people = set(chunk.people_mentioned) & set(other_chunk.people_mentioned)
            # Check for shared topics
            shared_topics = set(chunk.primary_topics) & set(other_chunk.primary_topics)
            # Check for cross-references
            is_referenced = chunk.date in other_chunk.cross_references
            
            if shared_people or shared_topics or is_referenced:
                related.append(other_chunk)
        
        return related

# Example usage:
def create_sample_chunk():
    chunk = JournalChunk(
        date=datetime.now(),
        content="Sample journal entry...",
        chunk_id="A1",
        word_count=150,
        size_category=ChunkSize.NORMAL,
        text_metrics=TextMetrics(
            word_count=150,
            sentence_count=10,
            avg_sentence_length=15.0,
            avg_word_length=4.5,
            reading_level=65.0,
            unique_words=100,
            vocabulary_richness=0.67
        ),
        sentiment_metrics=SentimentMetrics(
            overall_sentiment=Sentiment.POSITIVE,
            emotional_intensity=0.7,
            key_emotions=["joy", "anticipation"],
            confidence_score=0.85
        )
    )
    return chunk
