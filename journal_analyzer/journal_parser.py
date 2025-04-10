import re
from datetime import datetime
from typing import List, Tuple, Optional
import logging

from .config import JOURNAL_INPUT_FILE, DATE_FORMATS, DATE_HEADER_REGEX

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_date_string(date_str: str) -> Optional[datetime]:
    """Attempts to parse a date string using predefined formats."""
    # Clean up potential extra whitespace
    date_str = date_str.strip()
    for fmt in DATE_FORMATS:
        try:
            # Attempt to parse, return on first success
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue # Try next format
    # If no format matched
    logging.warning(f"Could not parse date string: '{date_str}' with any known format.")
    return None

def split_journal_entries(file_path: str = JOURNAL_INPUT_FILE) -> List[Tuple[datetime.date, str]]:
    """
    Splits the journal text file into entries based on date headers.

    Args:
        file_path: Path to the journal text file.

    Returns:
        A list of tuples, where each tuple contains (entry_date, entry_content).
        Entries with unparseable dates are skipped.
    """
    entries = []
    current_date = None
    current_content = []
    date_pattern = re.compile(DATE_HEADER_REGEX, re.IGNORECASE | re.MULTILINE)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logging.error(f"Journal file not found: {file_path}")
        return []
    except Exception as e:
        logging.error(f"Error reading journal file {file_path}: {e}")
        return []

    logging.info(f"Starting parsing of {file_path}...")
    line_num = 0
    for line in lines:
        line_num += 1
        match = date_pattern.match(line)
        if match:
            # Found a potential date header
            potential_date_str = match.group(0).strip()
            parsed_date = parse_date_string(potential_date_str)

            if parsed_date:
                # Successfully parsed a date, this marks a new entry
                # Save the previous entry if it exists
                if current_date and current_content:
                    entries.append((current_date, "".join(current_content).strip()))
                    logging.debug(f"Completed entry for {current_date}")

                # Start the new entry
                current_date = parsed_date.date() # Store only the date part
                current_content = [] # Reset content for the new entry
                logging.debug(f"Found new entry date: {current_date} on line {line_num}")
                # Don't add the date line itself to the content
                continue # Move to the next line

            else:
                # Matched regex but couldn't parse - treat as content
                logging.warning(f"Line {line_num} matched date regex but failed parsing: '{line.strip()}' - treating as content.")
                if current_date: # Only add if we are already inside an entry
                    current_content.append(line)

        elif current_date:
            # This line is part of the current entry's content
            current_content.append(line)
        else:
            # Line before the first valid date header - skip or log if needed
            logging.debug(f"Skipping line {line_num} before first valid date: '{line.strip()}'")


    # Add the last entry after the loop finishes
    if current_date and current_content:
        entries.append((current_date, "".join(current_content).strip()))
        logging.debug(f"Completed last entry for {current_date}")

    logging.info(f"Successfully parsed {len(entries)} entries from {file_path}.")
    return entries

# Example usage (for testing)
if __name__ == "__main__":
    parsed_entries = split_journal_entries()
    if parsed_entries:
        print(f"Parsed {len(parsed_entries)} entries.")
        print("\nFirst 5 Entry Dates:")
        for entry_date, _ in parsed_entries[:5]:
            print(entry_date)
        print("\nLast 5 Entry Dates:")
        for entry_date, _ in parsed_entries[-5:]:
            print(entry_date)
    else:
        print("No entries parsed.") 