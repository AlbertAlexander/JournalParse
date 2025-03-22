import spacy
import enchant
import json
import argparse
from pathlib import Path
from typing import Set, Dict, Tuple

class NameRedactor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        # Increase max length limit if needed
        self.nlp.max_length = 10000000
        self.replacements = {}
        # Initialize English dictionary
        self.dictionary = enchant.Dict("en_US")
        
    def detect_names(self, text: str) -> Set[str]:
        """Detect potential proper names using multiple heuristics"""
        names = set()
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Find named entities labeled as persons
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                # Split multi-word names
                for name in ent.text.split():
                    if self._is_likely_name(name):
                        names.add(name)
        
        # Additional pass for capitalized words that might be names
        for token in doc:
            if (token.text.istitle() and  # Capitalized
                len(token.text) > 1 and   # More than one letter
                not token.is_stop and     # Not a stopword
                not token.like_num and    # Not a number
                self._is_likely_name(token.text, token)):
                names.add(token.text)
        
        return names
    
    def _is_likely_name(self, word: str, token=None) -> bool:
        """Additional checks to validate if a word is likely a name"""
        # Skip common false positives
        skip_words = {
            'I', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
            'Saturday', 'Sunday', 'January', 'February', 'March', 'April',
            'May', 'June', 'July', 'August', 'September', 'October',
            'November', 'December', 'The', 'A', 'An', 'But', 'Or', 'And'
        }
        
        # Check if word is sentence initial
        is_sentence_start = False
        if token:
            # Get previous token if it exists
            prev_token = token.i > 0 and token.doc[token.i - 1]
            # Check if previous token ends a sentence or if this is the first token
            is_sentence_start = (
                not prev_token or  # First token in document
                isinstance(prev_token, str) and prev_token in '.!?' or  # Previous token is sentence ender
                getattr(prev_token, 'text', '') in '.!?'  # Previous spaCy token is sentence ender
            )
        
        # For sentence-initial words, check if it's a common word
        if is_sentence_start:
            # Check if the lowercase version is in the dictionary
            # Skip if it's a common word
            if self.dictionary.check(word.lower()):
                return False
        
        return (
            word.istitle() and                # Must be capitalized
            len(word) > 1 and                 # More than one letter
            word not in skip_words and        # Not a common false positive
            not word.isupper() and           # Not all caps (likely acronym)
            not any(c.isdigit() for c in word) and  # No numbers
            not any(c in word for c in '.,!?-/\\')  # No punctuation
        )
    
    def redact_names(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Redact names while maintaining consistency"""
        # Detect names
        names = self.detect_names(text)
        redacted = text
        
        # Generate replacements for new names
        for name in names:
            if name not in self.replacements:
                self.replacements[name] = f"[Name{len(self.replacements)+1}]"
        
        # Apply replacements
        for name, replacement in sorted(self.replacements.items(), key=lambda x: len(x[0]), reverse=True):
            redacted = redacted.replace(name, replacement)
        
        return redacted, self.replacements
    
    def save_replacements(self, output_path: str):
        """Save replacements to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.replacements, f, indent=2)
            
    def load_replacements(self, json_path: str):
        """Load existing replacements from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            self.replacements = json.load(f)

def process_file(input_path: str, mode: str = 'detect', replacements_path: str = None):
    """Process a file either to detect names or apply replacements"""
    redactor = NameRedactor()
    
    # Read input file
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    if mode == 'detect':
        # Detect names and generate replacements
        redacted_text, replacements = redactor.redact_names(text)
        
        # Save replacements to JSON
        output_json = Path(input_path).stem + '_replacements.json'
        redactor.save_replacements(output_json)
        
        # Save redacted text
        output_text = Path(input_path).stem + '_redacted.txt'
        with open(output_text, 'w', encoding='utf-8') as f:
            f.write(redacted_text)
            
        print(f"\nFound {len(replacements)} names:")
        for original, replacement in replacements.items():
            print(f"{original} -> {replacement}")
            
    elif mode == 'replace':
        if not replacements_path:
            raise ValueError("Replacements JSON file required for replace mode")
            
        # Load existing replacements
        redactor.load_replacements(replacements_path)
        
        # Apply replacements
        redacted = text
        for original, replacement in sorted(redactor.replacements.items(), 
                                         key=lambda x: len(x[0]), 
                                         reverse=True):
            redacted = redacted.replace(original, replacement)
            
        # Save redacted text
        output_text = Path(input_path).stem + '_redacted.txt'
        with open(output_text, 'w', encoding='utf-8') as f:
            f.write(redacted)

def main():
    parser = argparse.ArgumentParser(description='Process text files for name detection and replacement')
    parser.add_argument('input_file', help='Path to input text file')
    
    # Use mutually exclusive group for detect/replace
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--detect', action='store_true', 
                      help='Detect names and generate replacements JSON')
    group.add_argument('--replace', metavar='REPLACEMENTS_JSON',
                      help='Apply replacements from specified JSON file')
    
    args = parser.parse_args()
    
    if args.detect:
        process_file(args.input_file, mode='detect')
    else:
        process_file(args.input_file, mode='replace', replacements_path=args.replace)

if __name__ == "__main__":
    main() 