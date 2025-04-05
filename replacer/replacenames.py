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
    """Replace terms while preserving case and tracking replacements."""
    replacement_counts = defaultdict(int)
    all_terms = extract_mappings(mappings)
    
    # Sort by length (longest first) to prevent partial matches
    all_terms.sort(key=lambda x: len(x[0]), reverse=True)
    
    result = text
    for term, replacement, category in all_terms:
        if not term.strip() or not replacement.strip():
            continue
            
        pattern = create_replacement_pattern(term)
        
        def replace_match(match):
            matched_text = match.group(0)
            # Extract any possessive suffix
            possessive = ''
            if matched_text.endswith("'s"):
                possessive = "'s"
            elif matched_text.endswith("s'"):
                possessive = "s'"
                
            # Get leading punctuation/space
            leading = matched_text[:-len(term)-len(possessive)] if len(matched_text) > len(term) else ''
            
            # Count this replacement
            replacement_counts[(term, replacement, category)] += 1
            
            # Match case of the found text
            found_text = matched_text[len(leading):-len(possessive)] if possessive else matched_text[len(leading):]
            if found_text.isupper():
                return leading + replacement.upper() + possessive
            else:
                return leading + replacement + possessive

        result = re.sub(pattern, replace_match, result)
    
    return result, replacement_counts

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
