from anthropic import Anthropic, APIError, APITimeoutError, RateLimitError
import os
from typing import Dict, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential
from config import ANTHROPIC_API_KEY
import json
from jsonschema import validate, ValidationError
import re

class JournalAnalyzer:
    def __init__(self):
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Add the prompts dictionary
        # Consider implementing Roberta for emotions analysis
        self.prompts = {
            "imagery": """Analyze this journal entry for imagery and metaphors. 
            Only include imagery that seems important to the speaker, and only 
            include metaphors where a comparison is explicitly being made between 
            one thing and another. If an extended metaphor or a set of metaphors 
            relate to the same idea, group them together. For each image/metaphor 
            found, provide a list of objects in this exact format: 
            {{ 
                "metaphors": [ 
                    {{ 
                        "core_metaphor": "string describing the core metaphor, 
                        including the most essential concept or image", 
                        "raw_text": "exact text from the entry", 
                        "description": "what this metaphor is describing", 
                        "context": "emotional state or situation being referenced" 
                    }}
                ]
            }}
            Entry: {text}""",
            
            "emotions": """Analyze the emotional content of this journal entry on a continuous 0.0-10.0 scale, using one decimal place for precision. Provide analysis in this exact format:
            {{
                "emotional_dimensions": {{
                    "positive valence": X.X, // neutral (0.0) to extremely positive (10.0)
                    "negative valence": X.X, // positive (0.0) to extremely negative (-10.0)
                    "arousal": X.X, // calm/low energy (0.0) to excited/high energy (10.0)
                    "control": X.X, // feeling powerless (0.0) to empowered/in control (10.0)
                    "anxiety": X.X, // feeling completely at ease (0.0) to overwhelmingly anxious (10.0)
                    "confusion": X.X, // knowing exactly what to do (0.0) to completely lost (10.0)
                    "connection": X.X, // feeling isolated (0.0) to deep sense of belonging (10.0)
                    "social_activity": X.X, // solitary reflection/processing (0.0) to active external social engagement (10.0)
                }}
            }}
            Where X.X represents a number between 0.0 and 10.0 with one decimal place.

            Entry: {text}""",
        }
        
        # JSON schemas for validation
        self.schemas = {
            "imagery": {
                "type": "object",
                "properties": {
                    "metaphors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "core_metaphor": {"type": "string"},
                                "raw_text": {"type": "string"},
                                "description": {"type": "string"},
                                "context": {"type": "string"}
                            },
                            "required": ["core_metaphor", "raw_text", "description", "context"]
                        }
                    }
                },
                "required": ["metaphors"]
            },
            "emotions": {
                "type": "object",
                "properties": {
                    "emotional_dimensions": {
                        "type": "object",
                        "properties": {
                            "valence": {"type": "number", "minimum": 0.0, "maximum": 10.0},
                            "arousal": {"type": "number", "minimum": 0.0, "maximum": 10.0},
                            "control": {"type": "number", "minimum": 0.0, "maximum": 10.0}
                        },
                        "required": ["valence", "arousal", "control"]
                    }
                },
                "required": ["emotional_dimensions"]
            }
        }

    def validate_response(self, response: str, analysis_type: str) -> Dict:
        """Validate and parse Claude's response"""
        try:
            # Parse JSON response
            parsed = json.loads(response)
            
            # Validate against schema
            validate(instance=parsed, schema=self.schemas[analysis_type])
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {str(e)}")
            print(f"Raw response: {response}")
            raise
            
        except ValidationError as e:
            print(f"Response failed validation: {str(e)}")
            print(f"Raw response: {response}")
            raise
    
    def process_response(self, response: str, analysis_type: str) -> Dict:
        """Process Claude's response with specific handling for common formats"""
        result = {
            "raw_response": response,
            "parsed_data": None,
            "status": "success"
        }
        
        try:
            # Handle empty or invalid responses
            if not response or '{' not in response:
                result["status"] = "no_json_found"
                return result
            
            # Extract JSON content
            json_str = ""
            if "```json" in response:
                # Extract content between ```json and ```
                pattern = r"```json\n(.*?)\n```"
                matches = re.findall(pattern, response, re.DOTALL)
                if matches:
                    json_str = matches[0]
            else:
                # Find content between first { and last }
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = response[start:end]
            
            if not json_str:
                result["status"] = "no_json_found"
                return result
            
            # Clean the JSON string
            # 1. Remove comments (both // and multi-line)
            json_str = re.sub(r'//.*?\n|/\*.*?\*/', '', json_str, flags=re.DOTALL)
            
            # 2. Remove trailing commas before closing braces/brackets
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            
            # 3. Clean up whitespace and newlines
            json_str = '\n'.join(line.strip() for line in json_str.splitlines() if line.strip())
            
            # 4. Ensure proper JSON structure
            if not (json_str.startswith('{') and json_str.endswith('}')):
                raise json.JSONDecodeError("Malformed JSON", json_str, 0)
            
            # Parse the cleaned JSON
            parsed = json.loads(json_str)
            
            # Validate against schema
            if analysis_type in self.schemas:
                validate(instance=parsed, schema=self.schemas[analysis_type])
            
            result["parsed_data"] = parsed
            
        except json.JSONDecodeError as e:
            result["status"] = "json_parse_failed"
            print(f"JSON Parse Error in entry: {str(e)}\nProblematic JSON:\n{json_str}")
            
        except ValidationError as e:
            result["status"] = "schema_validation_failed"
            print(f"Schema Validation Error: {str(e)}")
            
        except Exception as e:
            result["status"] = f"error: {str(e)}"
            print(f"Processing Error: {str(e)}")
            
        return result

    def analyze_entry(self, text: str, analysis_type: str = "imagery") -> Dict:
        """Analyze a journal entry"""
        try:
            message = self.client.messages.create(
                model="claude-3-7-sonnet-20250219",  # Required
                max_tokens=4096,                    # Required
                messages=[{
                    "role": "user",
                    "content": self.prompts[analysis_type].format(text=text)
                }]
            )
            
            return self.process_response(message.content[0].text, analysis_type)
            
        except Exception as e:
            print(f"Error calling Claude: {str(e)}")
            return {
                "raw_response": None,
                "parsed_data": None,
                "status": f"api_error: {str(e)}"
            }

    def process_batch(self, entries: list, batch_size: int = 10) -> list:
        """Process multiple entries with rate limiting"""
        results = []
        for i in range(0, len(entries), batch_size):
            batch = entries[i:i + batch_size]
            batch_results = []
            for entry in batch:
                result = self.analyze_entry(entry)
                if result:
                    batch_results.append(result)
            results.extend(batch_results)
        return results
