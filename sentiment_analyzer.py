from textblob import TextBlob  # Simple option
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # Better for social text
import nltk
from transformers import pipeline  # Most sophisticated option
from typing import Dict, List, Union, Tuple
from models import JournalChunk, Sentiment, SentimentMetrics  # Add this import

class SentimentAnalyzer:
    def __init__(self, method: str = "vader"):
        """
        Initialize sentiment analyzer with chosen method
        method options: "textblob", "vader", "transformers"
        """
        self.method = method
        if method == "vader":
            self.analyzer = SentimentIntensityAnalyzer()
        elif method == "transformers":
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
        # TextBlob doesn't need initialization
        
        # For more granular emotion detection
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )

    def _convert_to_sentiment_enum(self, score: float) -> Sentiment:
        """Convert a normalized score (-1 to 1) to Sentiment enum"""
        if score <= -0.6:
            return Sentiment.VERY_NEGATIVE
        elif score <= -0.2:
            return Sentiment.NEGATIVE
        elif score <= 0.2:
            return Sentiment.NEUTRAL
        elif score <= 0.6:
            return Sentiment.POSITIVE
        else:
            return Sentiment.VERY_POSITIVE

    def analyze_chunk(self, chunk: JournalChunk) -> SentimentMetrics:
        """Analyze sentiment of a chunk using selected method"""
        text = chunk.content
        
        if self.method == "textblob":
            blob = TextBlob(text)
            normalized_score = blob.sentiment.polarity  # Already -1 to 1
            intensity = abs(normalized_score)
            emotions = ["neutral"]  # TextBlob doesn't provide emotion labels
            
        elif self.method == "vader":
            scores = self.analyzer.polarity_scores(text)
            normalized_score = scores["compound"]  # Already -1 to 1
            intensity = max(scores["neg"], scores["pos"])
            emotions = ["negative"] if scores["neg"] > scores["pos"] else ["positive"]
            
        elif self.method == "transformers":
            sentiment = self.analyzer(text)[0]
            emotions_result = self.emotion_analyzer(text)[0]
            
            # Convert transformer output to -1 to 1 scale
            if sentiment["label"] == "POSITIVE":
                normalized_score = sentiment["score"]
            else:
                normalized_score = -sentiment["score"]
                
            intensity = abs(normalized_score)
            
            # Get top 3 emotions
            emotions = sorted(
                emotions_result,
                key=lambda x: x["score"],
                reverse=True
            )[:3]
            emotions = [e["label"] for e in emotions]

        return SentimentMetrics(
            overall_sentiment=self._convert_to_sentiment_enum(normalized_score),
            emotional_intensity=min(1.0, intensity),  # Ensure 0-1 scale
            key_emotions=emotions,
            confidence_score=0.8  # Could be refined based on method
        )

    def analyze_entry_with_subchunks(self, chunk: JournalChunk, subchunks: List[JournalChunk] = None) -> Dict:
        """
        Analyze sentiment for an entry and its subchunks
        
        Args:
            chunk: The main journal chunk to analyze
            subchunks: Optional list of subchunks associated with the main chunk
        """
        results = {
            "chunk_id": chunk.chunk_id,
            "main_sentiment": self.analyze_chunk(chunk)
        }
        
        # If this is a subchunk, don't process further
        if chunk.sub_chunk_index is not None:
            return results
            
        # If we have subchunks, analyze each and aggregate
        if subchunks:
            subchunk_sentiments = [self.analyze_chunk(subchunk) for subchunk in subchunks]
            results["subchunk_sentiments"] = subchunk_sentiments
            results["aggregate_sentiment"] = self.aggregate_sentiments(subchunk_sentiments)
            
        return results

    def aggregate_sentiments(self, sentiments: List[SentimentMetrics]) -> SentimentMetrics:
        """Aggregate sentiments from subchunks"""
        if self.method == "vader":
            negative_intensity = sum(s.emotional_intensity for s in sentiments if s.overall_sentiment == Sentiment.NEGATIVE) / len(sentiments)
            neutral_intensity = sum(s.emotional_intensity for s in sentiments if s.overall_sentiment == Sentiment.NEUTRAL) / len(sentiments)
            positive_intensity = sum(s.emotional_intensity for s in sentiments if s.overall_sentiment == Sentiment.POSITIVE) / len(sentiments)
            overall_sentiment = Sentiment.NEGATIVE if negative_intensity > positive_intensity else Sentiment.POSITIVE if positive_intensity > negative_intensity else Sentiment.NEUTRAL
            emotional_intensity = max(negative_intensity, neutral_intensity, positive_intensity)
            key_emotions = ["negative"] if negative_intensity > positive_intensity else ["positive"] if positive_intensity > negative_intensity else ["neutral"]
        else:
            # Add aggregation for other methods as needed
            overall_sentiment = Sentiment.NEUTRAL
            emotional_intensity = 0.0
            key_emotions = ["neutral"]

        return SentimentMetrics(
            overall_sentiment=overall_sentiment,
            emotional_intensity=emotional_intensity,
            key_emotions=key_emotions,
            confidence_score=0.8  # Could be refined based on method
        )
