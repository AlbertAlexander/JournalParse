# Proposed Features

## 1. Smart Prompt Management
### Prompt Similarity Detection
- Use embeddings to detect similar previous prompts
- Compare new prompt with historical prompts using semantic similarity
- Suggest reuse of existing analyses when similarity exceeds threshold
- Option to replace or keep both versions

Example workflow:
```python
User: "How does the author express feelings of joy?"
AI: This prompt is 92% similar to previous query:
    "What expressions of happiness appear in the writing?"
    [analyzed on 2024-01-15]

Would you like to:
1. View previous analysis
2. Replace previous analysis
3. Continue with new analysis
4. Compare differences in prompts
```

Implementation considerations:
- Store embeddings alongside prompts
- Configurable similarity threshold
- Track prompt evolution over time
- Handle multiple similar matches

## 2. Hybrid MPNA Implementation
### Overview
Convert the current SQLite structure into a hybrid system that adds graph and vector capabilities while maintaining existing functionality.

### Required Components

#### A. Vector Extension for SQLite
- Add vector similarity search for prompts and content
- Implementation options:
  1. SQLite vector extension (sqlite-vss)
  2. Separate vector store (e.g., FAISS, Milvus)
  ```sql
  -- Example with sqlite-vss
  CREATE VIRTUAL TABLE embeddings USING vss0(
      embedding(384),    -- Vector dimension
      id INTEGER,        -- Reference to nodes
      node_type TEXT    -- 'prompt', 'entry', 'analysis'
  );
  ```

#### B. Graph Structure Tables
```sql
-- Core graph structure
CREATE TABLE nodes (
    node_id INTEGER PRIMARY KEY,
    node_type TEXT,
    content_ref TEXT,    -- Reference to original content
    properties TEXT      -- JSON metadata
);

CREATE TABLE edges (
    from_node INTEGER,
    to_node INTEGER,
    edge_type TEXT,
    weight REAL,
    properties TEXT,
    PRIMARY KEY (from_node, to_node, edge_type)
);

-- Maintain existing tables as content store
-- entries, llm_analysis_results, etc. remain unchanged
```

#### C. Graph Traversal Functions
Custom SQLite functions needed:
```sql
-- Find connected analyses
CREATE FUNCTION find_connected_analyses(node_id INTEGER)
RETURNS TABLE AS
WITH RECURSIVE traverse AS (
    SELECT node_id, 0 as depth
    FROM nodes
    WHERE node_id = ?
    UNION ALL
    SELECT e.to_node, t.depth + 1
    FROM traverse t
    JOIN edges e ON t.node_id = e.from_node
    WHERE t.depth < 3  -- Configurable depth
);
```

### Migration Path
1. **Phase 1: Add Graph Structure**
   - Keep existing tables
   - Add node/edge tables
   - Create migration script:
   ```python
   def migrate_to_graph():
       """Convert existing relationships to graph edges."""
       # Entries → nodes
       # Analysis results → nodes
       # Create edges for relationships
   ```

2. **Phase 2: Add Vector Search**
   - Add embedding storage
   - Generate embeddings for:
     - Prompts
     - Entry content
     - Analysis results

3. **Phase 3: Enhanced Queries**
   ```python
   def find_related_analyses(query: str):
       """Find analyses through graph traversal."""
       # 1. Find similar content via vector search
       # 2. Traverse graph for connected analyses
       # 3. Return combined results
   ```

### Benefits
1. **Richer Analysis**
   - Find indirect relationships
   - Pattern detection across entries
   - Semantic similarity search

2. **Better Context**
   - Graph traversal for related content
   - Temporal pattern detection
   - Connected analysis discovery

3. **Performance**
   - Efficient similarity search
   - Fast graph traversal
   - Maintains SQLite simplicity

### Technical Requirements
1. **Dependencies**
   ```text
   sqlite-vss
   faiss-cpu (alternative)
   sentence-transformers
   networkx (for graph algorithms)
   ```

2. **Storage**
   - Increased disk space for embeddings
   - Graph index structures
   - Vector indexes

3. **Processing**
   - Batch embedding generation
   - Graph maintenance
   - Regular index optimization

### Limitations
1. SQLite constraints on:
   - Concurrent access
   - Large-scale graph operations
   - Vector computation

2. Migration complexity:
   - Data consistency during migration
   - Performance tuning needed
   - Backup strategy required

## 3. [Future Feature]
...

Would you like me to expand on the technical implementation details for the prompt similarity feature? 