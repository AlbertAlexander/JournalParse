import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

def load_mappings(mapping_file: str) -> Dict[str, Any]:
    """Load and validate mappings from JSON file."""
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        return mappings
    except json.JSONDecodeError as e:
        print(f"Error in JSON file: {e}")
        print("Attempting to fix common JSON issues...")
        # Try to load with more permissive parsing
        with open(mapping_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Fix common JSON issues
            content = re.sub(r',\s*}', '}', content)  # Remove trailing commas
            content = re.sub(r',\s*]', ']', content)  # Remove trailing commas in arrays
            return json.loads(content)

def extract_mappings(data: Any, prefix: str = '') -> List[Tuple[str, str, str]]:
    """
    Recursively traverse JSON structure:
    1. Enter each category
    2. Process all key/value pairs in that category
    3. Move to next category
    4. Continue until all categories and nested categories are processed
    """
    mappings = []
    
    if not isinstance(data, dict):
        return mappings
        
    # Process each category/subcategory
    for key, value in data.items():
        current_category = f"{prefix}/{key}" if prefix else key
        
        if isinstance(value, dict):
            if value:  # If dictionary has content
                # First, get mappings from this subcategory
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, str):
                        # Direct key/value pair found
                        mappings.append((subkey, subvalue, current_category))
                    elif isinstance(subvalue, dict):
                        # Nested structure found, recurse into it
                        mappings.extend(extract_mappings(subvalue, current_category))
        elif isinstance(value, str):
            # Direct key/value pair found at this level
            mappings.append((key, value, prefix if prefix else 'default'))
    
    return mappings

def create_replacement_pattern(term: str) -> str:
    """
    Create regex pattern with strict word boundaries and possessive handling.
    Matches:
        - Exact term with word boundaries
        - Term with singular possessive ('s)
    Does not match:
        - Term as part of longer word
        - Term with plural possessive (s')
        - Term with other suffixes
    
    Warning:
        Plural possessives (e.g., "Toms'") are not replaced to avoid false positives
        and incorrect handling of actual plurals. Only singular possessives (e.g., "Tom's")
        are replaced.
    """
    escaped_term = re.escape(term)
    
    # Pattern components:
    # \b          - Word boundary (position between \w and \W chars)
    # {term}      - The exact term (case-sensitive)
    # (?:'s|s')? - Optional possessive: 's or s' (? makes the group optional)
    # \b          - Word boundary
    return rf'\b{escaped_term}(?:\'s|s\')?\b'

def replace_terms(text: str, mappings: Dict[str, Any]) -> Tuple[str, Dict[Tuple[str, str, str], int]]:
    """Replace terms while preserving case and tracking replacements.
    Processes one term at a time, longest first, to prevent partial matches.
    
    Case Handling:
        - Only replaces terms that match case exactly (e.g., "Tom" won't match "tom")
        - Replacement text preserves case of matched term:
            "Tom" -> "[Name1]"
            "TOM" -> "[NAME1]"
            "tom" -> "[name1]"
        - Non-matching cases are skipped to prevent incorrect replacements
    
    Possessive Handling:
        - Singular possessives ('s) are preserved:
            "Tom's" -> "[Name1]'s"
            "TOM'S" -> "[NAME1]'s"
        - Plural possessives (s') are NOT replaced:
            "Toms'" remains "Toms'"
        This prevents false positives with actual plurals and maintains
        grammatical correctness in the output text.
    
    Returns:
        Tuple of (processed_text, replacement_counts)
    """
    replacement_counts = defaultdict(int)
    all_terms = extract_mappings(mappings)
    all_terms.sort(key=lambda x: len(x[0]), reverse=True)  # Longest first
    
    result = text
    total_terms = len(all_terms)
    for term_idx, (term, replacement, category) in enumerate(all_terms, 1):
        if not term.strip():
            continue
            
        print(f"Processing term {term_idx}/{total_terms}: '{term}'")
        replacements_for_term = 0
        
        current_pos = 0
        while True:
            # Case-sensitive find
            pos = result.find(term, current_pos)
            if pos == -1:
                break
                
            # Get word with more context on each side (3 chars for safety)
            start = max(0, pos - 3)
            end = min(len(result), pos + len(term) + 3)
            word = result[start:end]
            
            # Since find() is case sensitive, we can simplify word boundary check
            if is_valid_word_match(word, term):
                # No need for case checks anymore - we know it matches exactly
                next_chars = result[pos+len(term):pos+len(term)+2]
                if next_chars == "'s":
                    matched = result[pos:pos+len(term)+2]
                    replacement_text = replacement + "'s"
                else:
                    matched = result[pos:pos+len(term)]
                    replacement_text = replacement
                
                result = result[:pos] + replacement_text + result[pos+len(matched):]
                replacement_counts[(term, replacement, category)] += 1
                replacements_for_term += 1
                current_pos = pos + len(replacement_text)
            else:
                current_pos = pos + 1
        
        if replacements_for_term > 0:
            print(f"Term {term_idx}/{total_terms}: '{term}' -> {replacements_for_term} replacements")
    
    return result, replacement_counts

def is_valid_word_match(word: str, term: str) -> bool:
    """Check if term is a valid word match (not part of a larger word)."""
    pos = word.find(term)  # Can be case sensitive now
    before = word[:pos]
    after = word[pos+len(term):]
    
    return (not before or not before[-1].isalnum()) and (not after or not after[0].isalnum())

def write_stats(stats_file: str, replacement_counts: Dict[Tuple[str, str, str], int]):
    """Write detailed statistics to file."""
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Replacement Statistics:\n\n")
        
        # Group by category
        by_category = defaultdict(list)
        for (term, replacement, category), count in replacement_counts.items():
            by_category[category].append((term, replacement, count))
        
        # Write statistics for each category
        for category in sorted(by_category.keys()):
            f.write(f"\n{category.upper()}:\n")
            f.write("-" * 40 + "\n")
            
            # Sort by count (highest first)
            items = sorted(by_category[category], key=lambda x: x[2], reverse=True)
            
            for term, replacement, count in items:
                if count == 0:
                    f.write(f"{term} -> {replacement}: NOT FOUND\n")
                else:
                    f.write(f"{term} -> {replacement}: {count} replacements\n")

def process_file(input_file: str, output_file: str, mapping_file: str, stats_file: str):
    """Process file and generate statistics."""
    try:
        mappings = load_mappings(mapping_file)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        processed_text, replacement_counts = replace_terms(text, mappings)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(processed_text)
        
        write_stats(stats_file, replacement_counts)
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Replace names and places in text files using mapping file.')
    parser.add_argument('input', help='Input text file path')
    parser.add_argument('output', help='Output text file path')
    parser.add_argument('mapping', help='JSON mapping file path')
    
    args = parser.parse_args()
    
    # Generate stats file name based on output file
    stats_file = Path(args.output).with_suffix('.stats.txt')
    
    try:
        process_file(args.input, args.output, args.mapping, stats_file)
        print(f"Processing completed successfully! Statistics written to {stats_file}")
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1
    return 0

if __name__ == "__main__":
    main()
