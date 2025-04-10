import sqlite3
from typing import Optional, List, Dict
import logging
from .database_manager import get_db_connection

def get_or_create_entity(name: str, entity_type: str) -> Optional[int]:
    """Finds an entity by name or creates it if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT entity_id FROM entities WHERE name = ?", (name,))
        result = cursor.fetchone()
        if result:
            return result['entity_id']
        
        cursor.execute("INSERT INTO entities (name, type) VALUES (?, ?)", 
                      (name, entity_type))
        conn.commit()
        entity_id = cursor.lastrowid
        logging.info(f"Created entity '{name}' (Type: {entity_type}) ID: {entity_id}")
        return entity_id
    except sqlite3.Error as e:
        logging.error(f"Error with entity '{name}': {e}")
        conn.rollback()
        return None
    finally:
        conn.close()

def link_entry_entity(entry_id: int, entity_id: int, snippet: Optional[str] = None):
    """Creates a link between an entry and an entity."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO entry_entities (entry_id, entity_id, context_snippet)
        VALUES (?, ?, ?)
        ON CONFLICT(entry_id, entity_id) DO NOTHING;
        """, (entry_id, entity_id, snippet))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error linking entry {entry_id} to entity {entity_id}: {e}")
        conn.rollback()
    finally:
        conn.close()

def get_entity_mentions(entity_id: int) -> List[Dict]:
    """Get all mentions of an entity with context."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
        SELECT e.entry_date, ee.context_snippet 
        FROM entry_entities ee
        JOIN entries e ON ee.entry_id = e.entry_id
        WHERE ee.entity_id = ?
        ORDER BY e.entry_date
        """, (entity_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
