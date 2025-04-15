from typing import Dict, Optional
import json
import logging
from .llm_manager import query_llm, parse_llm_json_response
from .config import DEFAULT_LLM_MODEL
from .error_manager import log_error

class LLMEmotionAnalyzer:
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """Analyze emotional content using LLM."""
        logging.info("Starting emotion analysis for entry")
        
        prompt = f"""You are an expert at analyzing emotional content in journal entries.
        Analyze the emotional state of the writer of this journal entry. Consider the full context,
        subtext, and nuanced emotional expressions.

        Provide two scores:
        1. Valence (positivity-negativity) on a 0-10 scale where:
           - 0 = extremely negative (even if brief)
           - 5 = neutral
           - 10 = extremely positive (even if brief)

        2. Arousal (emotional intensity/energy) on a 0-10 scale where:
           - 0 = completely calm/flat
           - 5 = moderate emotional energy
           - 10 = extremely high emotional intensity

        Also identify the primary emotions present and any emotional patterns. 
        Statements which do not contain emotional content or imply an emotional state will not affect the scores. 
        Scores reflect the state of the writer, not overall text.

        Journal text:
        {text}

        Respond in JSON format:
        {{
            "valence": float,
            "arousal": float,
            "primary_emotions": [str],
            "emotional_patterns": str,
            "confidence": float,
            "reasoning": str
        }}
        """

        try:
            response = query_llm(prompt)
            
            required_fields = ['valence', 'arousal', 'primary_emotions', 'emotional_patterns', 'confidence']
            defaults = {
                'valence': 5.0,
                'arousal': 5.0,
                'primary_emotions': ['neutral'],
                'emotional_patterns': 'No patterns detected',
                'confidence': 0.5,
                'reasoning': 'Incomplete analysis'
            }
            
            return parse_llm_json_response(response, required_fields, defaults)
            
        except Exception as e:
            logging.error(f"Error in emotion_analyzer: {e}")
            return None

    def analyze_emotional_development(self, entries: list[str]) -> Dict:
        """Analyze emotional patterns across multiple entries."""
        logging.info("Starting emotional development analysis")
        
        entries_text = "\n---\n".join(entries)
        
        prompt = f"""
        Analyze the emotional development and patterns across these journal entries.
        Focus on:
        1. How emotions evolve over time
        2. Recurring emotional patterns
        3. Emotional self-awareness and regulation
        4. Key emotional triggers

        Journal entries:
        {entries_text}

        Respond in JSON format:
        {{
            "emotional_trajectory": str,
            "recurring_patterns": [str],
            "growth_areas": [str],
            "key_triggers": [str],
            "recommendations": [str]
        }}
        """

        try:
            response = query_llm(prompt)
            
            required_fields = ['emotional_trajectory', 'recurring_patterns', 'growth_areas', 
                             'key_triggers', 'recommendations']
            defaults = {
                'emotional_trajectory': 'No clear trajectory detected',
                'recurring_patterns': [],
                'growth_areas': [],
                'key_triggers': [],
                'recommendations': []
            }
            
            return parse_llm_json_response(response, required_fields, defaults)
            
        except Exception as e:
            logging.error(f"Error in emotional development analysis: {e}")
            return None

# Example usage:
if __name__ == "__main__":
    analyzer = LLMEmotionAnalyzer()
    
    sample_text = """I am still feeling a lot of internal pressure to construct 
    and commit to a relationship model, not just a mode of interaction but an 
    identity and a philosophy too. A few days ago I advised myself to philosophize 
    less and love more. I'd expand that statement now to say, "Let your philosophy 
    and identity grow out of your choices. You can love Andrea in your way."""
    
    result = analyzer.analyze_emotion(sample_text)
    if result:
        print(json.dumps(result, indent=2)) 