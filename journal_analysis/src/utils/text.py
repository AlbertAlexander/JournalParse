import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import textstat

# review this to understand. why nltk? textstat usage?
# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def preprocess_text(text):
    """Clean and normalize text for analysis"""
    # Remove excess whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def count_words(text):
    """Count words in text"""
    if not text:
        return 0
    return len(word_tokenize(text))

def count_sentences(text):
    """Count sentences in text"""
    if not text:
        return 0
    return len(sent_tokenize(text))

def count_paragraphs(text):
    """Count paragraphs in text"""
    if not text:
        return 0
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    return len(paragraphs)

def calculate_reading_level(text):
    """Calculate Flesch-Kincaid grade level"""
    if not text or len(text) < 100:  # Need sufficient text
        return 0
    return textstat.flesch_kincaid_grade(text)

def count_pronouns(text):
    """Count personal pronouns in text"""
    # Define pronoun patterns
    patterns = {
        'I': r'\b[Ii]\b',
        'me': r'\b[Mm]e\b',
        'my': r'\b[Mm]y\b',
        'mine': r'\b[Mm]ine\b',
        'we': r'\b[Ww]e\b',
        'us': r'\b[Uu]s\b',
        'our': r'\b[Oo]ur\b',
        'ours': r'\b[Oo]urs\b',
        'you': r'\b[Yy]ou\b',
        'your': r'\b[Yy]our\b',
        'yours': r'\b[Yy]ours\b',
        'he': r'\b[Hh]e\b',
        'him': r'\b[Hh]im\b',
        'his': r'\b[Hh]is\b',
        'she': r'\b[Ss]he\b',
        'her': r'\b[Hh]er\b',
        'hers': r'\b[Hh]ers\b',
        'they': r'\b[Tt]hey\b',
        'them': r'\b[Tt]hem\b',
        'their': r'\b[Tt]heir\b',
        'theirs': r'\b[Tt]heirs\b'
    }
    
    counts = {}
    for pronoun, pattern in patterns.items():
        counts[pronoun] = len(re.findall(pattern, text))
    
    return counts

def extract_pseudonymized_references(text):
    """Extract pseudonymized references ([Name123], [Place456], etc.) from text"""
    name_pattern = r'\[Name\d+\]'
    place_pattern = r'\[Place\d+\]'
    business_pattern = r'\[Business\d+\]'
    
    names = re.findall(name_pattern, text)
    places = re.findall(place_pattern, text)
    businesses = re.findall(business_pattern, text)
    
    # Count occurrences
    name_counts = Counter(names)
    place_counts = Counter(places)
    business_counts = Counter(businesses)
    
    return {
        "names": [{"id": name, "count": count} for name, count in name_counts.items()],
        "places": [{"id": place, "count": count} for place, count in place_counts.items()],
        "businesses": [{"id": business, "count": count} for business, count in business_counts.items()]
    }

def find_sentences_with_reference(text, reference):
    """Find sentences containing a specific reference"""
    sentences = sent_tokenize(text)
    return [sent for sent in sentences if reference in sent]

def get_basic_text_stats(text):
    """Get comprehensive text statistics"""
    if not text:
        return {
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "reading_level": 0,
            "pronoun_counts": {},
            "references": {"names": [], "places": [], "businesses": []}
        }
    
    return {
        "word_count": count_words(text),
        "sentence_count": count_sentences(text),
        "paragraph_count": count_paragraphs(text),
        "reading_level": calculate_reading_level(text),
        "pronoun_counts": count_pronouns(text),
        "references": extract_pseudonymized_references(text)
    }

def chunk_text_for_llm(text, chunk_size=3000, overlap=200):
    """Split text into chunks for LLM processing with overlap between chunks"""
    chunks = []
    
    if not text or len(text) <= chunk_size:
        return [text] if text else []
    
    # Split text by paragraphs
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    for para in paragraphs:
        # If adding this paragraph exceeds chunk size and we already have content
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Keep some overlap from the previous chunk
            last_sentences = sent_tokenize(current_chunk)
            overlap_text = ""
            
            # Add sentences from the end until we reach desired overlap
            for sent in reversed(last_sentences):
                if len(overlap_text) + len(sent) <= overlap:
                    overlap_text = sent + " " + overlap_text
                else:
                    break
            
            current_chunk = overlap_text
        
        current_chunk += para + "\n\n"
    
    # Add the final chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks
