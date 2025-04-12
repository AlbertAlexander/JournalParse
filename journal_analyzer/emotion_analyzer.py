from typing import Dict, Optional
import json
import logging
from .llm_manager import query_llm  # Assuming you have this set up
from .config import DEFAULT_LLM_MODEL
from .error_manager import log_error

class LLMEmotionAnalyzer:
    def __init__(self, model: str = DEFAULT_LLM_MODEL):
        self.model = model
        
    def analyze_emotion(self, entry_id: int, text: str) -> Dict[str, float]:
        """Analyze emotional content using LLM."""
        
        prompt = f"""You are an expert at analyzing emotional content in journal entries.
        Analyze the emotional content of this journal entry. Consider the full context,
        subtext, and nuanced emotional expressions.

        Provide two scores:
        1. Valence (positivity-negativity) on a 0-10 scale where:
           - 0 = extremely negative
           - 5 = neutral
           - 10 = extremely positive

        2. Arousal (emotional intensity/energy) on a 0-10 scale where:
           - 0 = completely calm/flat
           - 5 = moderate emotional energy
           - 10 = extremely high emotional intensity

        Also identify the primary emotions present and any emotional patterns.

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
            response = query_llm(prompt, self.model)
            if not response:
                raise ValueError("No response from LLM")
            
            # Defensive JSON parsing
            try:
                # Extract JSON if it's embedded in other text
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found in response")
                
                # Validate required fields
                required_fields = ['valence', 'arousal', 'primary_emotions', 'emotional_patterns', 'confidence']
                missing_fields = [field for field in required_fields if field not in result]
                
                if missing_fields:
                    logging.warning(f"Missing fields in emotion analysis: {missing_fields}")
                    # Add default values for missing fields
                    defaults = {
                        'valence': 5.0,
                        'arousal': 5.0,
                        'primary_emotions': ['neutral'],
                        'emotional_patterns': 'No patterns detected',
                        'confidence': 0.5,
                        'reasoning': 'Incomplete analysis'
                    }
                    for field in missing_fields:
                        result[field] = defaults[field]
                
                # Validate and fix data types
                if not isinstance(result.get('primary_emotions', []), list):
                    result['primary_emotions'] = [str(result.get('primary_emotions', 'neutral'))]
                
                # Ensure primary_emotions is a list of strings
                result['primary_emotions'] = [str(emotion) for emotion in result.get('primary_emotions', [])]
                
                # Validate scores are in range
                result['valence'] = max(0, min(10, float(result.get('valence', 5.0))))
                result['arousal'] = max(0, min(10, float(result.get('arousal', 5.0))))
                result['confidence'] = max(0, min(1, float(result.get('confidence', 0.5))))
                
                # Ensure other fields are strings
                result['emotional_patterns'] = str(result.get('emotional_patterns', 'No patterns detected'))
                result['reasoning'] = str(result.get('reasoning', 'No reasoning provided'))
                
                return result
                
            except json.JSONDecodeError as e:
                logging.error(f"JSON parsing error in emotion analysis: {e}")
                # Return a default result
                return {
                    'valence': 5.0,
                    'arousal': 5.0,
                    'primary_emotions': ['neutral'],
                    'emotional_patterns': 'Error parsing LLM response',
                    'confidence': 0.0,
                    'reasoning': f"JSON parsing error: {str(e)}"
                }
                
        except Exception as e:
            log_error(
                analysis_type='entry_emotion',
                error=e,
                entry_id=entry_id,
                context={'text_length': len(text)}
            )
            return None

    def analyze_emotional_development(self, entries: list[str]) -> Dict:
        """Analyze emotional patterns across multiple entries."""
        
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
            return json.loads(query_llm(prompt, self.model))
        except Exception as e:
            logging.error(f"Error in emotional development analysis: {e}")
            return {"error": str(e)}

# Example usage:
if __name__ == "__main__":
    analyzer = LLMEmotionAnalyzer()
    
    sample_text = """I am still feeling a lot of internal pressure to construct 
    and commit to a relationship model, not just a mode of interaction but an 
    identity and a philosophy too. A few days ago I advised myself to philosophize 
    less and love more. I'd expand that statement now to say, "Let your philosophy 
    and identity grow out of your choices. You can love Andrea in your way."""
    
    result = analyzer.analyze_emotion(1, sample_text)
    if result:
        print(json.dumps(result, indent=2)) 