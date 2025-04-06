from dateutil import parser as date_parser
import re
from pathlib import Path
from src.utils.text import chunk_text_for_llm, preprocess_text

def parse_journal_file(filepath):
    """Parse raw journal file into entry objects."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Pattern to identify entries (customize to your journal format)
    entry_pattern = r'(?:^|\n)(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\n(.*?)(?=\n\d{4}-\d{2}-\d{2}|\n\d{2}/\d{2}/\d{4}|\Z)'
    
    entries = []
    for match in re.finditer(entry_pattern, text, re.DOTALL):
        date_str, content = match.groups()
        
        # Parse date to standardized format
        try:
            date_obj = date_parser.parse(date_str)
            date = date_obj.strftime('%Y-%m-%d')
            entry_id = date.replace('-', '')
        except:
            continue
        
        # Clean and preprocess the content
        clean_content = preprocess_text(content.strip())
        
        entries.append({
            "id": entry_id,
            "date": date,
            "raw_text": clean_content,
            "chunks": chunk_text_for_llm(clean_content)
        })
    
    return entries

def chunk_text(text, chunk_size=3000):
    """Split text into chunks for LLM processing."""
    chunks = []
    current_chunk = ""
    
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        # If adding this paragraph exceeds chunk size, save current chunk
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = ""
        
        current_chunk += para + "\n\n"
    
    # Add final chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks
