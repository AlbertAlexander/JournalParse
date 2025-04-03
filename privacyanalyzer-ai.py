import json
from pathlib import Path
import subprocess
from typing import Dict, List, Set, Tuple
from tqdm import tqdm
import time
from datetime import datetime
import re
import argparse
import sys
import ollama

class OllamaClient:
    """Handles Ollama interactions - either local or remote"""
    def __init__(self, mode='local', host='localhost', port=11434):
        self.mode = mode
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        
    def generate(self, model: str, prompt: str, options: dict = None) -> dict:
        """Unified generation method with minimal response"""
        default_options = {
            'temperature': 0,
            'raw': True  # Add raw option to get minimal response
        }
        if options:
            default_options.update(options)
        
        if self.mode == 'local':
            try:
                import ollama
                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options=default_options
                )
                # Strip context if still present
                return {
                    'response': response.get('response', ''),
                    'done': response.get('done', True)
                }
            except ImportError:
                print("Ollama library not found. Falling back to API...")
                self.mode = 'remote'
            except Exception as e:
                print(f"Local Ollama error: {e}")
                print("Falling back to API...")
                self.mode = 'remote'
        
        if self.mode == 'remote':
            try:
                import requests
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": default_options
                    }
                )
                data = response.json()
                # Strip context if present
                return {
                    'response': data.get('response', ''),
                    'done': data.get('done', True)
                }
            except Exception as e:
                print(f"Remote Ollama error: {e}")
                raise

class ContextualPseudonymizer:
    def __init__(self, model: str = "llama3.3:latest", mode='local', host='localhost', port=11434):
        """Initialize with configurable Ollama client"""
        self.model = model
        self.ollama_client = OllamaClient(mode=mode, host=host, port=port)
        self.output_dir = None  # Add this to store output directory
        # Initialize empty substitutions dictionary
        self.substitutions = {
            'names': {},
            'places': {},
            'contacts': {},
            'businesses': {},
            'dates': {},
            'relationships': {}
        }
        self.runtime_substitutions = {}  # Track substitutions made during this run
        self.identified_terms = set()
        self.preserve = {'cities', 'states', 'countries', 'months', 'days_of_week'}
        self.identifier_occurrences = {}
        
        # Verify Ollama server is running and check model
        try:
            # Get list of models
            response = ollama.list()
            print("✓ Connected to Ollama server")
            
            # Debug the response
            print("Available models:", response)
            
            # Check if our model exists
            model_name = self.model.split(':')[0]  # Handle cases like 'qwen2.5:3b'
            if not any(m.get('name', '').startswith(model_name) for m in response.get('models', [])):
                print(f"Model {self.model} not found. Pulling...")
                ollama.pull(self.model)
                print(f"✓ Successfully pulled {self.model}")
            else:
                print(f"✓ Model {self.model} is available")
            
        except ConnectionError:
            print("✗ Could not connect to Ollama server")
            print("Please start the server with 'ollama serve' in a terminal")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error connecting to Ollama: {e}")
            print("Full error:", str(e))
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    def _run_ollama(self, prompt: str) -> str:
        """Execute Ollama command with better streaming and timeout handling"""
        try:
            print("Sending query to Ollama...")
            
            process = subprocess.Popen(
                ["ollama", "run", self.model, "--nowordwrap", prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Initialize response collection
            response_chunks = []
            empty_chunks = 0
            print("Starting to collect response chunks...")  # Debug print
            
            while True:
                # Read with timeout using select
                import select
                reads, _, _ = select.select([process.stdout, process.stderr], [], [], 1.0)
                
                if not reads:
                    empty_chunks += 1
                    print(".", end="", flush=True)
                    # If no data for 30 seconds, assume something's wrong
                    if empty_chunks > 30:
                        print("\nNo response for 30 seconds, terminating...")
                        process.kill()
                        break
                    continue
                
                empty_chunks = 0
                
                for reader in reads:
                    chunk = reader.readline()
                    if chunk:
                        if reader == process.stderr:
                            print(f"\nStderr: {chunk.strip()}")
                        else:
                            response_chunks.append(chunk)
                            print("✓", end="", flush=True)
                
                # Check if process has finished
                if process.poll() is not None:
                    break
            
            # Get any remaining output
            remaining_out, remaining_err = process.communicate()
            if remaining_out:
                print("\nGot remaining output")  # Debug print
                response_chunks.append(remaining_out)
            if remaining_err:
                print(f"\nFinal stderr: {remaining_err}")
            
            print(f"\nTotal chunks collected: {len(response_chunks)}")  # Debug print
            
            if response_chunks:
                print("\n✓ Received response from Ollama")
                response_text = "".join(response_chunks)
                print(f"Response length: {len(response_text)}")  # Debug print
                
                try:  # Add try/except around logging
                    log_dir = self.output_dir / "ollama_logs"
                    print(f"Creating log directory: {log_dir}")  # Debug print
                    log_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_file = log_dir / f"ollama_response_{timestamp}.txt"
                    print(f"Writing to log file: {log_file}")  # Debug print
                    with open(log_file, "w") as f:
                        f.write(f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response_text}")
                    print("Log file written successfully")  # Debug print
                except Exception as e:
                    print(f"Error writing log file: {e}")  # Debug error
                
                return response_text
            else:
                print("\n✗ No response chunks received from Ollama")  # Changed message
                return ""
            
        except Exception as e:
            print(f"\n✗ Error running Ollama: {e}")
            return ""

    def detect_identifiers(self, text: str, chunk_index: int = None) -> Dict:
        """Use Ollama API to detect identifiers in text"""
        try:
            print(f"Processing chunk of {len(text):,} chars")
            print("\nSample text:", text[:200], "...")
            
            prompt = f"""Analyze this text for personal identifying information. Extract all names, places, contacts, businesses, and dates. Include every occurrence.

TEXT:
{text}

Return ONLY this JSON structure listing ALL occurrences. Use this EXACT format, including ALL found items::
{{
    "names": {{
            "names": ["ALL person names, first or last, exactly as they appear in the text"],
            "role_identifiers": ["ANY roles like 'Dr.', 'Professor', 'Officer', 'Wife' with associated names"]
    }},
    "places": {{
        "addresses": ["EXACT addresses as written"],
        "landmarks": ["EXACT place names"],
        "neighborhoods": ["EXACT area names"]
    }},
    "contacts": {{
        "phones": ["ONLY actual phone numbers"],
        "emails": ["ONLY actual email addresses"],
        "social_media": ["ONLY actual usernames"]
    }},
    "businesses": {{
        "specific_businesses": ["EXACT business names"],
        "institutions": ["EXACT organization names"]
    }}
}}

IMPORTANT: 
- Include ALL found items
- Only identify proper nouns as names (like "John" or "Sarah")
- Do NOT include pronouns (like I, me, my, we, us, you, he, she, they, them)
- Do NOT include general terms (friend, person, partner, someone)
- Do not return empty arrays unless nothing was found
- Return items exactly as they appear in text
- Do not include any explanatory text, ONLY the JSON object with the extracted data"""

            try:
                # Switch to API approach
                response = self.ollama_client.generate(
                    model=self.model,
                    prompt=prompt
                )
                
                try:
                    # Clean the response to ensure we only parse the JSON
                    json_start = response['response'].find('{')
                    json_end = response['response'].rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response['response'][json_start:json_end]
                        
                        try:
                            result = json.loads(json_str)
                            
                            # Check for missing categories or subcategories
                            issues = []
                            expected_categories = {
                                "names": ["names", "role_identifiers"],
                                "places": ["addresses", "landmarks", "neighborhoods"],
                                "contacts": ["phones", "emails", "social_media"],
                                "businesses": ["specific_businesses", "institutions"]
                            }
                            
                            # Check for missing categories
                            for category, subcategories in expected_categories.items():
                                if category not in result:
                                    issues.append(f"Missing category: {category}")
                                    # Add empty category
                                    result[category] = {}
                                
                                # Check for missing subcategories
                                if category in result:
                                    for subcategory in subcategories:
                                        if subcategory not in result[category]:
                                            issues.append(f"Missing subcategory: {category}.{subcategory}")
                                            # Add empty subcategory
                                            result[category][subcategory] = []
                            
                            # Log issues if any
                            if issues and chunk_index is not None:
                                error_log_dir = self.output_dir / "error_logs"
                                error_log_dir.mkdir(exist_ok=True)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                error_file = error_log_dir / f"json_structure_error_{chunk_index+1}_{timestamp}.txt"
                                with open(error_file, "w") as f:
                                    f.write(f"CHUNK: {chunk_index+1}\n\n")
                                    f.write("STRUCTURE ISSUES:\n")
                                    for issue in issues:
                                        f.write(f"- {issue}\n")
                                    f.write("\nRAW RESPONSE:\n")
                                    f.write(response['response'])
                                    f.write("\n\nPROCESSED JSON:\n")
                                    f.write(json.dumps(result, indent=2))
                                
                                print(f"\n⚠️ Found {len(issues)} structure issues in chunk {chunk_index+1}. See log for details.")
                            
                            # Process and display categories
                            print("\nProcessing categories:")
                            for category, subcategories in result.items():
                                print(f"Category: {category}")
                                print(f"Subcategories: {subcategories}")
                                for subcategory, items in subcategories.items():
                                    print(f"  Subcategory: {subcategory}")
                                    print(f"  Items: {items}")
                            
                            # Log after all processing is complete
                            log_dir = self.output_dir / "ollama_logs"
                            log_dir.mkdir(exist_ok=True)
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            log_file = log_dir / f"ollama_response_{timestamp}.txt"
                            with open(log_file, "w") as f:
                                f.write(f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response['response']}\n\nPROCESSED:\n{json.dumps(result, indent=2)}")
                            
                            return result
                        except KeyError as e:
                            print(f"\nMissing key in JSON: {str(e)}")
                            if chunk_index is not None:
                                self._log_json_error(chunk_index, f"KeyError: {str(e)}", response['response'], json_str)
                            return {}
                        except Exception as e:
                            print(f"\nError processing JSON: {str(e)}")
                            if chunk_index is not None:
                                self._log_json_error(chunk_index, f"Processing error: {str(e)}", response['response'], json_str)
                            return {}
                    else:
                        print("No valid JSON found in response")
                        if chunk_index is not None:
                            self._log_json_error(chunk_index, "No valid JSON found", response['response'])
                        return {}
                    
                except json.JSONDecodeError as e:
                    print(f"\nJSON parsing error: {str(e)}")
                    if chunk_index is not None:
                        self._log_json_error(chunk_index, f"JSON decode error: {str(e)}", response['response'])
                    return {}
                    
            except Exception as e:
                print(f"Generation error: {e}")
                if chunk_index is not None:
                    self._log_json_error(chunk_index, f"Generation error: {str(e)}", "No response received")
                return {}
            
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            if chunk_index is not None:
                self._log_json_error(chunk_index, f"Unexpected error: {str(e)}", "")
            return {}

    def _log_json_error(self, chunk_index, error_message, response, json_str=None):
        """Log JSON errors to file for debugging"""
        try:
            error_log_dir = self.output_dir / "error_logs"
            error_log_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = error_log_dir / f"json_error_{chunk_index+1}_{timestamp}.txt"
            
            with open(error_file, "w") as f:
                f.write(f"CHUNK: {chunk_index+1}\n")
                f.write(f"ERROR: {error_message}\n\n")
                f.write("RAW RESPONSE:\n")
                f.write(response)
                
                if json_str:
                    f.write("\n\nEXTRACTED JSON STRING:\n")
                    f.write(json_str)
            
            print(f"Error details logged to {error_file}")
            
        except Exception as e:
            print(f"Error logging JSON error: {e}")

    def generate_contextual_pseudonym(self, original: str, category: str, 
                                    context: str = "") -> str:
        """Generate meaningful pseudonym that preserves context"""
        if original in self.substitutions[category]:
            return self.substitutions[category][original]
            
        prompt = f"""Generate a privacy-preserving replacement for this {category} that preserves similar meaning/context.
Original: {original}
Context: {context}
Requirements:
- Maintain similar structure/format
- Preserve general meaning
- Change identifying details
- Be consistent with typical {category}
Return only the replacement, no other text."""
        
        pseudonym = self._run_ollama(prompt).strip()
        self.substitutions[category][original] = pseudonym
        return pseudonym

    def _chunk_text(self, text: str, chunk_size: int = 10000) -> List[str]:
        """Split text into ~10k char chunks, breaking at sentence boundaries"""
        chunks = []
        start = 0
        total_length = len(text)
        
        while start < total_length:
            # Get chunk of target size
            end = min(start + chunk_size, total_length)
            
            # If not at end of text, find a good sentence break
            if end < total_length:
                # Look for sentence endings (.!?) followed by space or newline
                for i in range(end, max(start, end - 200), -1):
                    if i < total_length and text[i-1] in '.!?' and (text[i].isspace() or text[i] == '\n'):
                        end = i
                        break
            
            chunk = text[start:end]
            chunks.append(chunk)
            print(f"Chunk {len(chunks)}: {len(chunk):,} chars")
            
            start = end
        
        print(f"Split {total_length:,} chars into {len(chunks)} chunks")
        return chunks

    def create_changelog(self):
        """Generate a changelog for the current state of substitutions"""
        changelog = ""
        for category, items in self.substitutions.items():
            for item, pseudonym in items.items():
                changelog += f"Replaced {item} with {pseudonym} in {category}\n"
        return changelog

    def _load_existing_substitutions(self, output_dir: Path, original_name: str) -> bool:
        """Load existing substitutions with error handling"""
        substitutions_file = output_dir / f"{original_name}_substitutions.json"
        try:
            if substitutions_file.exists():
                print(f"Found existing substitutions file: {substitutions_file}")
                with open(substitutions_file, 'r') as f:
                    existing = json.load(f)
                    # Update substitutions while preserving structure
                    for category in self.substitutions:
                        self.substitutions[category].update(existing.get(category, {}))
                    print(f"Loaded {sum(len(subs) for subs in self.substitutions.values())} existing substitutions")
                    return True
            else:
                print("No existing substitutions file found. Will create new substitutions.")
                return False
        except Exception as e:
            print(f"Error loading substitutions file: {e}")
            print("Continuing with empty substitutions...")
            return False

    def get_substitution(self, term: str, category: str, context: str = "") -> str:
        """Get substitution with fallback hierarchy"""
        # 1. Check existing substitutions from file
        if term in self.substitutions[category]:
            return self.substitutions[category][term]
        
        # 2. Check runtime substitutions
        if term in self.runtime_substitutions:
            return self.runtime_substitutions[term]
        
        # 3. Generate new substitution
        new_substitution = self.generate_contextual_pseudonym(term, category, context)
        
        # Store in both runtime and permanent dictionaries
        self.runtime_substitutions[term] = new_substitution
        self.substitutions[category][term] = new_substitution
        
        print(f"Created new substitution: {term} -> {new_substitution} ({category})")
        return new_substitution

    def _generate_substitutions(self):
        """Generate substitutions for identified terms with gender preservation"""
        for term, category in self.identified_terms:
            if term not in self.substitutions[category]:
                # Generate a substitution if one doesn't exist
                if category == 'names':
                    # Use index for unique numbering
                    idx = len(self.substitutions[category]) + 1
                    # Common gendered name patterns
                    male_patterns = ['ben', 'rogelio', 'will']  # Add more as needed
                    female_patterns = ['andrea']  # Add more as needed
                    
                    if any(term.lower() == p for p in male_patterns):
                        self.substitutions[category][term] = f"[MaleName{idx}]"
                    elif any(term.lower() == p for p in female_patterns):
                        self.substitutions[category][term] = f"[FemaleName{idx}]"
                    else:
                        self.substitutions[category][term] = f"[Name{idx}]"
                elif category == 'places':
                    self.substitutions[category][term] = f"[Place{len(self.substitutions[category])+1}]"
                elif category == 'contacts':
                    self.substitutions[category][term] = f"[Contact{len(self.substitutions[category])+1}]"
                elif category == 'businesses':
                    self.substitutions[category][term] = f"[Business{len(self.substitutions[category])+1}]"
                elif category == 'relationships':
                    self.substitutions[category][term] = f"[Relationship{len(self.substitutions[category])+1}]"

    def process_file(self, file_path: str):
        """Process file with additional error handling"""
        try:
            input_path = Path(file_path)
            if not input_path.exists():
                print(f"Error: File not found: {file_path}")
                sys.exit(1)

            self.output_dir = input_path.parent / "pseudonymized_output"
            self.output_dir.mkdir(exist_ok=True)
            
            # Create error log directory
            error_log_dir = self.output_dir / "error_logs"
            error_log_dir.mkdir(exist_ok=True)
            
            # Try to load existing substitutions first
            self._load_existing_substitutions(self.output_dir, input_path.stem)
            
            start_time = time.time()
            print(f"Reading {file_path}...")
            
            try:
                text = input_path.read_text()
                print(f"Successfully read file. Length: {len(text)} characters")
            except Exception as e:
                print(f"Error reading file: {e}")
                sys.exit(1)
            
            # First pass: identify terms
            print("\nFirst pass: Identifying terms...")
            chunks = self._chunk_text(text)
            print(f"Split text into {len(chunks)} chunks")
            
            # Track all terms found across chunks
            all_terms = set()
            failed_chunks = []

            try:
                for i, chunk in enumerate(tqdm(chunks, desc="Identifying terms")):
                    print(f"\nProcessing chunk {i+1}/{len(chunks)}")
                    try:
                        # Pass chunk index for better error logging
                        identifiers = self.detect_identifiers(chunk, i)
                        
                        if identifiers:  # Only process if we got valid identifiers
                            for category, subcategories in identifiers.items():
                                for subcategory, items in subcategories.items():
                                    for item in items:
                                        if item and not any(item.lower() in self.preserve for p in self.preserve):
                                            # Add to both running sets
                                            all_terms.add((item, category))
                                            self.identified_terms.add((item, category))
                                            key = (item, category)
                                            self.identifier_occurrences[key] = text.count(item)
                    except Exception as e:
                        # Log the error but continue processing
                        error_msg = f"Error processing chunk {i+1}: {str(e)}"
                        print(f"\n⚠️ {error_msg}")
                        failed_chunks.append((i, chunk[:100] + "...", str(e)))
                        
                        # Save error details to log
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        error_file = error_log_dir / f"chunk_error_{i+1}_{timestamp}.txt"
                        with open(error_file, "w") as f:
                            f.write(f"ERROR: {str(e)}\n\nCHUNK CONTENT:\n{chunk}")
                        
                        # Continue with next chunk
                        continue
                    
                    # Save progress periodically
                    if i % 5 == 0:  # Save every 5 chunks
                        # Generate substitutions before saving
                        self._generate_substitutions()
                        self._save_progress(self.output_dir, chunks[:i+1], i, input_path.stem)

            except KeyboardInterrupt:
                print("\n\nProcessing interrupted! Saving current progress...")
                self._save_progress(self.output_dir, chunks[:i+1], i, input_path.stem)
                print("Progress saved. You can resume from the last saved state.")
                sys.exit(1)
            
            # After all chunks, update identified_terms with everything found
            self.identified_terms.update(all_terms)
            self._generate_substitutions()  # Generate final substitutions
            self._save_progress(self.output_dir, chunks, len(chunks)-1, input_path.stem)
            
            # Log failed chunks summary
            if failed_chunks:
                print(f"\n⚠️ {len(failed_chunks)} chunks failed processing:")
                with open(error_log_dir / "failed_chunks_summary.txt", "w") as f:
                    f.write(f"Total failed chunks: {len(failed_chunks)}\n\n")
                    for idx, preview, error in failed_chunks:
                        print(f"  - Chunk {idx+1}: {error}")
                        f.write(f"Chunk {idx+1}: {preview}\nError: {error}\n\n")
            
            # Process chunks with consistent substitution handling
            print("Applying substitutions...")
            processed_chunks = []
            for i, chunk in enumerate(tqdm(chunks, desc="Pseudonymizing text")):
                processed_text = chunk
                
                # Sort terms by length (longest first)
                sorted_terms = sorted(self.identified_terms, key=lambda x: len(x[0]), reverse=True)
                
                for term, category in sorted_terms:
                    if term in processed_text:
                        replacement = self.get_substitution(term, category, chunk)
                        processed_text = re.sub(
                            re.escape(term),
                            replacement,
                            processed_text,
                            flags=re.IGNORECASE
                        )
                
                processed_chunks.append(processed_text)
                
                # Save progress periodically
                if i % 10 == 0:
                    self._save_progress(self.output_dir, processed_chunks, i, input_path.stem)

            # Save final results
            self._save_final_results(self.output_dir, processed_chunks, input_path.stem)
            
            # Generate detailed report
            self._save_detailed_report(self.output_dir, input_path.stem)
            
            print(f"\nProcessing completed in {(time.time() - start_time) / 60:.1f} minutes")
            print(f"New substitutions created: {len(self.runtime_substitutions)}")
            print(f"Total substitutions used: {sum(len(subs) for subs in self.substitutions.values())}")
            if failed_chunks:
                print(f"⚠️ {len(failed_chunks)} chunks failed processing. See error logs for details.")

        except Exception as e:
            print(f"\nError processing file: {e}")
            sys.exit(1)

    def _get_output_directory(self, output_dir: Path, original_name: str) -> Path:
        """Get or create appropriate output directory"""
        # Check for existing runs of this model on this file
        existing_dirs = sorted([
            d for d in output_dir.glob(f"*_{self.model}")
            if d.is_dir() and d.name.split('_')[1] == self.model
        ])
        
        if existing_dirs:
            latest_dir = existing_dirs[-1]
            # If last run was today, use that directory
            if latest_dir.name.startswith(datetime.now().strftime("%Y%m%d")):
                print(f"Continuing in existing directory: {latest_dir}")
                return latest_dir
        
        # Create new dated directory if no existing today
        date_str = datetime.now().strftime("%Y%m%d")
        new_dir = output_dir / f"{date_str}_{self.model}"
        new_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created new directory: {new_dir}")
        return new_dir

    def _save_progress(self, output_dir: Path, chunks: List[str], chunk_num: int, original_name: str):
        """Save intermediate results with version control"""
        model_dir = self._get_output_directory(output_dir, original_name)
        
        # Add run timestamp to track multiple passes in same day
        timestamp = datetime.now().strftime("%H%M%S")
        
        # Base filenames with pass number
        base_files = {
            'substitutions': model_dir / f"{original_name}_substitutions.json",
            'partial': model_dir / f"{original_name}_pseudonymized_partial.txt",
            'terms': model_dir / f"{original_name}_identified_terms.json",
            'log': model_dir / "processing_log.txt"
        }
        
        # Save current state
        with open(base_files['substitutions'], 'w') as f:
            json.dump(self.substitutions, f, indent=2)
        
        with open(base_files['partial'], 'w') as f:
            f.write('\n\n'.join(chunks[:chunk_num+1]))
        
        with open(base_files['terms'], 'w') as f:
            json.dump(list(self.identified_terms), f, indent=2)
        
        # Append to processing log
        with open(base_files['log'], 'a') as f:
            f.write(f"\n[{timestamp}] Processed chunk {chunk_num+1}/{len(chunks)}")
        
        print(f"\nSaved progress at chunk {chunk_num+1}/{len(chunks)} to {model_dir}")

    def _save_final_results(self, output_dir: str, processed_chunks: List[str], original_name: str):
        """Save final results and changelog"""
        # Save pseudonymized text
        with open(output_dir / f"{original_name}_pseudonymized.txt", 'w') as f:
            f.write('\n\n'.join(processed_chunks))
        
        # Save substitutions
        with open(output_dir / f"{original_name}_substitutions.json", 'w') as f:
            json.dump(self.substitutions, f, indent=2)
        
        # Save changelog
        with open(output_dir / f"{original_name}_changelog.txt", 'w') as f:
            f.write(self.create_changelog())
        
        # Generate statistics
        stats = {
            "total_substitutions": sum(len(subs) for subs in self.substitutions.values()),
            "categories": {
                category: len(subs) 
                for category, subs in self.substitutions.items()
            }
        }
        
        with open(output_dir / f"{original_name}_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)

    def _save_detailed_report(self, output_dir: str, original_name: str):
        """Generate detailed report including substitution sources"""
        report = "Substitution Report\n"
        report += "=================\n\n"
        
        # Existing substitutions
        report += "Reused Substitutions:\n"
        report += "------------------\n"
        reused = set(sum((list(subs.keys()) for subs in self.substitutions.values()), [])) - set(self.runtime_substitutions.keys())
        for term in sorted(reused):
            for category, subs in self.substitutions.items():
                if term in subs:
                    report += f"{term} -> {subs[term]} ({category})\n"
        
        # New substitutions
        report += "\nNew Substitutions:\n"
        report += "----------------\n"
        for term in sorted(self.runtime_substitutions.keys()):
            report += f"{term} -> {self.runtime_substitutions[term]}\n"
        
        # Statistics
        report += "\nStatistics:\n"
        report += "-----------\n"
        report += f"Total terms processed: {len(self.identified_terms)}\n"
        report += f"Reused substitutions: {len(reused)}\n"
        report += f"New substitutions: {len(self.runtime_substitutions)}\n"
        
        with open(output_dir / f"{original_name}_detailed_report.txt", 'w') as f:
            f.write(report)

    def _test_model(self):
        """Test if model is responding"""
        print("Testing model response...")
        test_response = self._run_ollama("Respond with 'ok' if you can read this.")
        if "ok" in test_response.lower():
            print("✓ Model is responding correctly")
            return True
        else:
            print("✗ Model is not responding as expected")
            print(f"Response received: {test_response}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Pseudonymize personal information in text while preserving context"
    )
    parser.add_argument(
        "file", 
        help="Path to the text file to process"
    )
    parser.add_argument(
        "--model", 
        default="llama3.3:latest",
        help="Ollama model to use (default: llama3.3:latest)"
    )
    # Add new connection arguments
    parser.add_argument(
        "--mode",
        choices=['local', 'remote'],
        default='local',
        help="Use local or remote Ollama (default: local)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Ollama host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=11434,
        help="Ollama port (default: 11434)"
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume processing from a specific chunk index (0-based)"
    )
    
    args = parser.parse_args()
    
    try:
        pseudonymizer = ContextualPseudonymizer(
            model=args.model,
            mode=args.mode,
            host=args.host,
            port=args.port
        )
        pseudonymizer.process_file(args.file)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()