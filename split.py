import re
from datetime import datetime
from models import JournalChunk, JournalCollection, TextMetrics, SentimentMetrics, Sentiment, ChunkSize
from enum import Enum
from typing import Optional, List, Dict
import anthropic
import os
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class ChunkSize(Enum):
    NORMAL = "normal"      # <= 1500 words
    LONG = "long"         # 1501-3000 words
    TOO_LONG = "too_long" # > 3000 words

def parse_date(date_string: str) -> datetime:
    """Convert various date formats to datetime object"""
    # Clean the date string by replacing dots with slashes
    date_string = date_string.replace('.', '/')
    
    # Try MM/DD/YY or MM/DD/YYYY format
    try:
        return datetime.strptime(date_string, '%m/%d/%y')
    except ValueError:
        try:
            return datetime.strptime(date_string, '%m/%d/%Y')
        except ValueError:
            # Try written format (e.g., "December 25, 2024")
            try:
                return datetime.strptime(date_string, '%B %d, %Y')
            except ValueError:
                # Try abbreviated month format (e.g., "Dec 25, 2024")
                try:
                    return datetime.strptime(date_string, '%b %d, %Y')
                except ValueError:
                    raise ValueError(f"Unable to parse date: {date_string}")

def analyze_text_metrics(text: str) -> TextMetrics:
    """Calculate text metrics for a chunk of text"""
    words = text.split()
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    return TextMetrics(
        word_count=len(words),
        sentence_count=len(sentences),
        avg_sentence_length=len(words) / len(sentences) if sentences else 0,
        avg_word_length=sum(len(w) for w in words) / len(words) if words else 0,
        reading_level=65.0,  # placeholder - implement proper Flesch-Kincaid
        unique_words=len(set(words)),
        vocabulary_richness=len(set(words)) / len(words) if words else 0
    )

def analyze_sentiment(text: str) -> SentimentMetrics:
    """Analyze sentiment of text"""
    # Placeholder - implement proper sentiment analysis
    return SentimentMetrics(
        overall_sentiment=Sentiment.NEUTRAL,
        emotional_intensity=0.5,
        key_emotions=[],
        confidence_score=0.8
    )

def get_chunk_size_category(word_count: int) -> ChunkSize:
    if word_count <= 1500:
        return ChunkSize.NORMAL
    elif word_count <= 3000:
        return ChunkSize.LONG
    else:
        return ChunkSize.TOO_LONG

def split_journal_into_chunks(text: str, max_words=1500, limit=None) -> JournalCollection:
    # Regular expression to match dates in format M/D/YY, MM/DD/YYYY, or M.D.Y, MM.DD.YYYY
    date_pattern = r'\b(?:1[0-2]|0?[1-9])(?:[/.])(?:3[01]|[12][0-9]|0?[1-9])(?:[/.])(?:20)?[0-9]{2}\b'
    
    # Also match written dates like "December 25, 2024" or "Dec 25, 2024"
    written_date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}\b'
    
    # Combine patterns
    combined_pattern = f"({date_pattern}|{written_date_pattern})"
    
    chunks = []
    chunk_count = 0
    current_chunk = ""
    current_date = None
    
    # Split text into lines
    lines = text.split('\n')
    raw_chunks = []
    
    for line in lines:
        # Look for date matches
        date_match = re.search(combined_pattern, line)
        
        if date_match:
            # If we have a previous chunk, save it
            if current_chunk and current_date:
                raw_chunks.append({
                    'date': current_date,
                    'content': current_chunk.strip()
                })
            
            # Start new chunk
            current_date = date_match.group(0)
            current_chunk = line + "\n"
        else:
            # Add line to current chunk
            current_chunk += line + "\n"
    
    # Add final chunk if exists
    if current_chunk and current_date:
        raw_chunks.append({
            'date': current_date,
            'content': current_chunk.strip()
        })
    
    # Process raw chunks into JournalChunk objects
    for i, raw_chunk in enumerate(raw_chunks):
        if limit and chunk_count >= limit:
            break
            
        # Validate date exists
        if not raw_chunk['date']:
            print(f"Warning: Skipping chunk {i+1} - no valid date found")
            continue
            
        try:
            parsed_date = parse_date(raw_chunk['date'])
        except ValueError as e:
            print(f"Warning: Skipping chunk {i+1} - {str(e)}")
            continue
            
        content = raw_chunk['content']
        word_count = len(content.split())
        size_category = get_chunk_size_category(word_count)
        
        if size_category == ChunkSize.TOO_LONG:
            words = content.split()
            for j in range(0, len(words), max_words):
                if limit and chunk_count >= limit:
                    break
                    
                sub_content = ' '.join(words[j:j + max_words])
                sub_chunk = JournalChunk(
                    date=parsed_date,
                    content=sub_content,
                    chunk_id=f"A{i+1}.{j//max_words + 1}",
                    word_count=len(sub_content.split()),
                    size_category=ChunkSize.NORMAL,
                    sub_chunk_index=j//max_words + 1,
                    total_sub_chunks=(len(words) + max_words - 1) // max_words,
                    text_metrics=analyze_text_metrics(sub_content),
                    sentiment_metrics=analyze_sentiment(sub_content)
                )
                chunks.append(sub_chunk)
                chunk_count += 1
        else:
            chunk = JournalChunk(
                date=parsed_date,
                content=content,
                chunk_id=f"A{i+1}",
                word_count=word_count,
                size_category=size_category,
                text_metrics=analyze_text_metrics(content),
                sentiment_metrics=analyze_sentiment(content)
            )
            chunks.append(chunk)
            chunk_count += 1
    
    if not chunks:
        raise ValueError("No valid journal entries found in the text")
        
    return JournalCollection(chunks=chunks)

def analyze_chunks(chunks, analysis_function):
    """
    Layer Alpha: Analyze each chunk independently
    """
    results = []
    for i, chunk in enumerate(chunks):
        result = analysis_function(chunk)
        results.append({
            'chunk_id': f'A{i+1}',
            'date': chunk['date'],
            'analysis': result
        })
    return results

def prepare_chunk_prompt(chunk: JournalChunk) -> str:
    """Prepare a prompt for analyzing a single chunk"""
    return f"""Analyze this journal entry from {chunk.date.strftime('%B %d, %Y')}:

Content:
{chunk.content}

Please provide:
1. Overall mood/sentiment
2. Key themes
3. Important events or people mentioned
4. Any cross-references to other times/events
"""

def synthesize_results(alpha_results: List[Dict], analysis_type: str) -> str:
    """
    Prepare a synthesis prompt for Claude to analyze patterns across chunks
    """
    chunks_summary = "\n\n".join([
        f"Entry {result['chunk_id']} ({result['date']}): {result['analysis']['summary']}"
        for result in alpha_results
    ])
    
    synthesis_prompt = f"""I have analyzed {len(alpha_results)} journal entries. Here are the individual analyses:

{chunks_summary}

Please provide a synthesis of these entries, focusing on:
1. Overall patterns and trends
2. Changes over time
3. Recurring themes
4. Cross-references between entries
5. Any retrospective insights where later entries reflect on earlier times

Analysis type requested: {analysis_type}
"""
    
    return synthesis_prompt

def process_with_claude(collection: JournalCollection, analysis_type: str) -> Dict:
    """
    Orchestrate the two-layer analysis process
    """
    # Layer Alpha: Individual chunk analysis
    alpha_results = []
    for chunk in collection.chunks:
        prompt = prepare_chunk_prompt(chunk)
        # Here you would send prompt to Claude and get response
        print(f"To analyze chunk {chunk.chunk_id}, send this prompt to Claude:\n{prompt}\n")
        
        # Store the results (you'll replace this with actual Claude response)
        alpha_results.append({
            'chunk_id': chunk.chunk_id,
            'date': chunk.date.isoformat(),
            'analysis': {'summary': 'Placeholder for Claude analysis'}
        })
    
    # Layer Beta: Synthesis
    synthesis_prompt = synthesize_results(alpha_results, analysis_type)
    print(f"\nFor synthesis, send this prompt to Claude:\n{synthesis_prompt}")
    
    return {
        'alpha_results': alpha_results,
        'synthesis_prompt': synthesis_prompt
    }

# Example usage:
def example_mood_analysis(chunk):
    """
    Example analysis function for mood analysis
    """
    # Implement sentiment analysis here
    # Return structured analysis of mood for the chunk
    pass

# Main processing
def process_journal(text, analysis_type):
    # Split into chunks
    chunks = split_journal_into_chunks(text)
    
    # Layer Alpha: Analyze each chunk
    alpha_results = analyze_chunks(chunks, example_mood_analysis)
    
    # Layer Beta: Synthesize overall analysis
    beta_result = synthesize_results(alpha_results, analysis_type)
    
    return {
        'alpha_results': alpha_results,
        'beta_synthesis': beta_result
    }

def process_test_sample(file_path: str, num_entries: int = 5) -> JournalCollection:
    """Process just a few entries for testing purposes"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    return split_journal_into_chunks(text, limit=num_entries)

class JournalAnalyzer:
    def __init__(self, api_key: str = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
        )
        self.model = "claude-3-sonnet-20240229"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def analyze_chunk(self, chunk: JournalChunk) -> Dict:
        """Analyze a single chunk using Claude"""
        prompt = self._prepare_chunk_prompt(chunk)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return {
                'chunk_id': chunk.chunk_id,
                'date': chunk.date.isoformat(),
                'analysis': response.content
            }
        except Exception as e:
            print(f"Error analyzing chunk {chunk.chunk_id}: {str(e)}")
            raise

    def synthesize_results(self, alpha_results: List[Dict], analysis_type: str) -> Dict:
        """Synthesize all chunk analyses using Claude"""
        synthesis_prompt = self._prepare_synthesis_prompt(alpha_results, analysis_type)
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": synthesis_prompt
                }]
            )
            return {
                'type': analysis_type,
                'synthesis': response.content,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error in synthesis: {str(e)}")
            raise

    def process_journal(self, collection: JournalCollection, analysis_type: str) -> Dict:
        """Process entire journal with rate limiting"""
        alpha_results = []
        
        for chunk in collection.chunks:
            try:
                result = self.analyze_chunk(chunk)
                alpha_results.append(result)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Failed to process chunk {chunk.chunk_id}: {str(e)}")
                continue
        
        synthesis = self.synthesize_results(alpha_results, analysis_type)
        
        return {
            'alpha_results': alpha_results,
            'beta_synthesis': synthesis
        }

    def _prepare_chunk_prompt(self, chunk: JournalChunk) -> str:
        """Prepare prompt for single chunk analysis"""
        return f"""Analyze this journal entry from {chunk.date.strftime('%B %d, %Y')}:

Content:
{chunk.content}

Please provide a JSON response with:
1. overall_mood: string (positive/negative/neutral)
2. sentiment_score: float (-1.0 to 1.0)
3. key_themes: list of strings
4. important_entities: list of people and places mentioned
5. cross_references: list of any references to other times or events
6. summary: brief analysis of the entry"""

    def _prepare_synthesis_prompt(self, alpha_results: List[Dict], analysis_type: str) -> str:
        """Prepare prompt for synthesis"""
        chunks_summary = "\n\n".join([
            f"Entry {result['chunk_id']} ({result['date']}): {result['analysis']}"
            for result in alpha_results
        ])
        
        return f"""I have analyzed {len(alpha_results)} journal entries. Here are the individual analyses:

{chunks_summary}

Please provide a JSON response synthesizing these entries with:
1. overall_patterns: list of major patterns observed
2. temporal_changes: description of changes over time
3. recurring_themes: list of themes that appear multiple times
4. cross_references: list of connections between entries
5. retrospective_insights: any cases where later entries reflect on earlier times
6. meta_analysis: overall assessment of the writer's journey

Analysis type: {analysis_type}"""
