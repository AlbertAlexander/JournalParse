from pathlib import Path
import json
import sqlite3
import os
from datetime import datetime

class JournalDB:
    def __init__(self, data_dir="data/journal_data"):
        """Initialize database connection and structure"""
        self.data_dir = Path(data_dir)
        self.entries_dir = self.data_dir / "entries"
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup SQLite connection
        self.db_path = self.data_dir / "journal.db"
        self.conn = sqlite3.connect(str(self.db_path))
        self._initialize_db()
        
        # Initialize index
        self.index_path = self.data_dir / "metadata.json"
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "entries_count": 0,
                "date_range": {"start": None, "end": None},
                "last_updated": None,
                "analyses_completed": []
            }
    
    def _initialize_db(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Main entries table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS entries (
            id TEXT PRIMARY KEY,
            date TEXT,
            word_count INTEGER,
            sentence_count INTEGER,
            reading_level REAL,
            sentiment_score REAL,
            arousal_score REAL
        )''')
        
        # References table for names, places, businesses
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS references (
            entry_id TEXT,
            ref_type TEXT,  /* name, place, business */
            ref_id TEXT,    /* [Name617], [Place257] */
            count INTEGER,
            FOREIGN KEY (entry_id) REFERENCES entries (id),
            PRIMARY KEY (entry_id, ref_type, ref_id)
        )''')
        
        # Pronoun usage table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pronoun_usage (
            entry_id TEXT,
            pronoun TEXT,
            count INTEGER,
            FOREIGN KEY (entry_id) REFERENCES entries (id),
            PRIMARY KEY (entry_id, pronoun)
        )''')
        
        # LLM analysis results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS llm_analysis (
            entry_id TEXT,
            analysis_type TEXT,
            json_data TEXT,
            completed_at TEXT,
            FOREIGN KEY (entry_id) REFERENCES entries (id),
            PRIMARY KEY (entry_id, analysis_type)
        )''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS date_index ON entries (date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS ref_type_index ON references (ref_type, ref_id)')
        self.conn.commit()
    
    def save_entry(self, entry_data):
        """Save a journal entry to both JSON and SQLite"""
        entry_id = entry_data.get("id")
        
        # Save to JSON file
        entry_path = self.entries_dir / f"{entry_id}.json"
        with open(entry_path, 'w', encoding='utf-8') as f:
            json.dump(entry_data, f, indent=2)
        
        # Extract data for SQLite
        cursor = self.conn.cursor()
        
        # Insert/update in entries table
        programmatic = entry_data.get("analysis", {}).get("programmatic", {})
        llm_analysis = entry_data.get("analysis", {}).get("llm", {})
        
        cursor.execute('''
        INSERT OR REPLACE INTO entries 
        (id, date, word_count, sentence_count, reading_level, sentiment_score, arousal_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry_id,
            entry_data.get("date"),
            programmatic.get("word_count", 0),
            programmatic.get("sentence_count", 0),
            programmatic.get("reading_level", 0),
            llm_analysis.get("sentiment", {}).get("positive", 0),
            llm_analysis.get("sentiment", {}).get("arousal", 0)
        ))
        
        # Save references (if available)
        if "references" in entry_data.get("analysis", {}):
            self._save_references(entry_id, entry_data)
        
        # Save pronoun counts (if available)
        if "pronoun_counts" in programmatic:
            self._save_pronoun_counts(entry_id, programmatic["pronoun_counts"])
        
        # Commit changes
        self.conn.commit()
        
        # Update metadata
        self._update_metadata(entry_data)
        
        return entry_id
    
    def _save_references(self, entry_id, entry_data):
        """Save references to names, places, businesses from entry"""
        cursor = self.conn.cursor()
        
        # Clear existing references for this entry
        cursor.execute("DELETE FROM references WHERE entry_id = ?", (entry_id,))
        
        # Get references
        references = entry_data.get("analysis", {}).get("references", {})
        
        # Save names
        for name_ref in references.get("names", []):
            cursor.execute('''
            INSERT INTO references (entry_id, ref_type, ref_id, count)
            VALUES (?, ?, ?, ?)
            ''', (entry_id, "name", name_ref, 1))
        
        # Save places
        for place_ref in references.get("places", []):
            cursor.execute('''
            INSERT INTO references (entry_id, ref_type, ref_id, count)
            VALUES (?, ?, ?, ?)
            ''', (entry_id, "place", place_ref, 1))
            
        # Save businesses
        for business_ref in references.get("businesses", []):
            cursor.execute('''
            INSERT INTO references (entry_id, ref_type, ref_id, count)
            VALUES (?, ?, ?, ?)
            ''', (entry_id, "business", business_ref, 1))
    
    def _save_pronoun_counts(self, entry_id, pronoun_counts):
        """Save pronoun usage statistics"""
        cursor = self.conn.cursor()
        
        # Clear existing pronoun counts for this entry
        cursor.execute("DELETE FROM pronoun_usage WHERE entry_id = ?", (entry_id,))
        
        # Insert new counts
        for pronoun, count in pronoun_counts.items():
            cursor.execute('''
            INSERT INTO pronoun_usage (entry_id, pronoun, count)
            VALUES (?, ?, ?)
            ''', (entry_id, pronoun, count))
    
    def _update_metadata(self, entry_data):
        """Update database metadata"""
        # Update entry count
        self.metadata["entries_count"] = len(os.listdir(self.entries_dir))
        
        # Update date range
        entry_date = entry_data.get("date")
        if entry_date:
            if self.metadata["date_range"]["start"] is None or entry_date < self.metadata["date_range"]["start"]:
                self.metadata["date_range"]["start"] = entry_date
            if self.metadata["date_range"]["end"] is None or entry_date > self.metadata["date_range"]["end"]:
                self.metadata["date_range"]["end"] = entry_date
        
        # Update last_updated timestamp
        self.metadata["last_updated"] = datetime.now().isoformat()
        
        # Save metadata file
        with open(self.index_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_entry(self, entry_id):
        """Retrieve a journal entry by ID"""
        entry_path = self.entries_dir / f"{entry_id}.json"
        if not entry_path.exists():
            return None
        
        with open(entry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_analysis(self, entry_id, analysis_type, analysis_data):
        """Update a specific analysis for an entry"""
        # Get the entry
        entry = self.get_entry(entry_id)
        if not entry:
            return False
        
        # Update the analysis data
        if "analysis" not in entry:
            entry["analysis"] = {}
            
        if analysis_type in ["programmatic", "llm", "references"]:
            entry["analysis"][analysis_type] = analysis_data
        else:
            # For specific LLM analysis types (like "sentiment", "metaphors")
            if "llm" not in entry["analysis"]:
                entry["analysis"]["llm"] = {}
            entry["analysis"]["llm"][analysis_type] = analysis_data
        
        # Save to JSON
        self.save_entry(entry)
        
        # For LLM analysis, also save to the llm_analysis table
        if analysis_type not in ["programmatic", "references"]:
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT OR REPLACE INTO llm_analysis
            (entry_id, analysis_type, json_data, completed_at)
            VALUES (?, ?, ?, ?)
            ''', (
                entry_id, 
                analysis_type, 
                json.dumps(analysis_data), 
                datetime.now().isoformat()
            ))
            self.conn.commit()
            
            # Update metadata
            if analysis_type not in self.metadata["analyses_completed"]:
                self.metadata["analyses_completed"].append(analysis_type)
                self._update_metadata(entry)
        
        return True
    
    def get_all_entries(self):
        """Get all entry IDs in chronological order"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM entries ORDER BY date")
        return [row[0] for row in cursor.fetchall()]
    
    def get_entries_by_date_range(self, start_date=None, end_date=None):
        """Get entries within a date range"""
        cursor = self.conn.cursor()
        
        query = "SELECT id FROM entries WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        cursor.execute(query, params)
        return [row[0] for row in cursor.fetchall()]
    
    def get_entries_with_reference(self, ref_id):
        """Get all entries that mention a specific reference"""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT DISTINCT e.id FROM entries e
        JOIN references r ON e.id = r.entry_id
        WHERE r.ref_id = ?
        ORDER BY e.date
        ''', (ref_id,))
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_top_references(self, ref_type="name", limit=100):
        """Get the most frequently mentioned references of a specific type"""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT ref_id, SUM(count) as total_count, COUNT(DISTINCT entry_id) as entries_count
        FROM references
        WHERE ref_type = ?
        GROUP BY ref_id
        ORDER BY total_count DESC
        LIMIT ?
        ''', (ref_type, limit))
        
        return [{"id": row[0], "count": row[1], "entries": row[2]} for row in cursor.fetchall()]
    
    def close(self):
        """Close the database connection"""
        self.conn.close()
