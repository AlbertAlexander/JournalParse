# Manual Database Operations Guide

## Setup
```bash
# Open SQLite database
sqlite3 data/journal_analysis.db

# Enable better output formatting
.mode column
.headers on
```

## Finding Issues

### View Recent Errors
```sql
-- Show last 10 errors
SELECT error_id, analysis_type, 
       substr(error_message, 1, 50) as error_preview,
       error_timestamp 
FROM analysis_errors 
WHERE resolved = FALSE 
ORDER BY error_timestamp DESC 
LIMIT 10;

-- Count errors by type
SELECT analysis_type, COUNT(*) as error_count 
FROM analysis_errors 
WHERE resolved = FALSE 
GROUP BY analysis_type;
```

### Inspect Raw Responses
```sql
-- Get full error context
SELECT error_details 
FROM analysis_errors 
WHERE error_id = <error_id>;

-- View temporal analysis raw response
SELECT analysis_id, time_period_start, time_period_end, llm_response 
FROM llm_analysis_results 
WHERE analysis_id = <analysis_id>;
```

## Manual Fixes

### Fix Entry-Level Emotion Analysis
```sql
-- View current emotion analysis
SELECT * FROM emotion_analysis WHERE entry_id = <entry_id>;

-- Update emotion scores
UPDATE emotion_analysis 
SET 
    valence_score = <value>,          -- 0-10 scale
    arousal_score = <value>,          -- 0-10 scale
    primary_emotions = '["emotion1", "emotion2"]',
    analysis_confidence = <value>,     -- 0-1 scale
    llm_reasoning = 'Manual correction: <reason>'
WHERE entry_id = <entry_id>;
```

### Fix Temporal Analysis
```sql
-- Update temporal analysis results
UPDATE llm_analysis_results 
SET llm_response = '{
    "analysis": "corrected analysis text",
    "key_patterns": ["pattern1", "pattern2"],
    "significant_periods": ["period1", "period2"],
    "confidence": 0.95
}'
WHERE analysis_id = <analysis_id>;
```

### Mark Errors as Resolved
```sql
-- Mark single error resolved
UPDATE analysis_errors 
SET 
    resolved = TRUE,
    resolution_timestamp = CURRENT_TIMESTAMP,
    resolution_notes = 'description of fix'
WHERE error_id = <error_id>;

-- Mark multiple related errors resolved
UPDATE analysis_errors 
SET 
    resolved = TRUE,
    resolution_timestamp = CURRENT_TIMESTAMP,
    resolution_notes = 'batch fix applied'
WHERE analysis_type = '<type>' 
AND error_message LIKE '%specific error%';
```

## Data Validation

### Check Entry Coverage
```sql
-- Find entries missing emotion analysis
SELECT e.entry_id, e.entry_date 
FROM entries e 
LEFT JOIN emotion_analysis ea ON e.entry_id = ea.entry_id 
WHERE ea.entry_id IS NULL;

-- Check temporal analysis coverage
SELECT DISTINCT strftime('%Y-%m', entry_date) as month,
       COUNT(*) as entry_count,
       EXISTS (
           SELECT 1 
           FROM llm_analysis_results lar 
           WHERE date(entry_date) BETWEEN lar.time_period_start AND lar.time_period_end
       ) as has_analysis
FROM entries 
GROUP BY month 
ORDER BY month;
```

### Data Quality Checks
```sql
-- Find suspicious emotion scores
SELECT entry_id, valence_score, arousal_score 
FROM emotion_analysis 
WHERE valence_score < 0 OR valence_score > 10 
   OR arousal_score < 0 OR arousal_score > 10;

-- Check for malformed JSON responses
SELECT analysis_id, llm_response 
FROM llm_analysis_results 
WHERE json_valid(llm_response) = 0;
```

## Database Maintenance

### Backup Database
```bash
# From command line
cp data/journal_analysis.db data/journal_analysis.backup.$(date +%Y%m%d_%H%M%S).db

# From within SQLite
.backup data/journal_analysis.backup.db
```

## Rerunning/Updating Analyses

### Find Exact Query
```sql
-- Find the exact query you want to rerun/modify
SELECT question_ref, time_period_start, time_period_end
FROM llm_analysis_results
WHERE question_ref LIKE '%keyword%'
ORDER BY analysis_timestamp DESC;
```

### Update Existing Analysis
```sql
-- Update using exact query match
UPDATE llm_analysis_results
SET llm_response = '{
    "analysis": "updated analysis",
    "key_points": ["point1", "point2"],
    "evidence": ["evidence1", "evidence2"],
    "confidence": 0.9
}'
WHERE question_ref = 'exact query text'
AND (time_period_start = '2023-01-01' OR time_period_start IS NULL)
AND (time_period_end = '2023-12-31' OR time_period_end IS NULL);
```
```

### Optimize Database
```sql
-- Optimize database size and performance
VACUUM;
ANALYZE;
```

## Tips
- Always make a backup before manual fixes
- Use transactions for multiple related updates:
  ```sql
  BEGIN TRANSACTION;
  -- make changes
  COMMIT;  -- or ROLLBACK if needed
  ```
- SQLite treats single quotes as string delimiters
- Use json_valid() to verify JSON formatting before updates
- Date format should be 'YYYY-MM-DD'

## Common JSON Templates

### Emotion Analysis
```json
{
    "valence": 7.5,
    "arousal": 4.2,
    "primary_emotions": ["joy", "contentment"],
    "emotional_patterns": "Description of patterns",
    "confidence": 0.9,
    "reasoning": "Explanation of analysis"
}
```

### Temporal Analysis
```json
{
    "analysis": "Overall period analysis",
    "key_patterns": [
        "Pattern description 1",
        "Pattern description 2"
    ],
    "significant_periods": [
        "Notable event/period 1",
        "Notable event/period 2"
    ],
    "overall_trajectory": "Development description",
    "confidence": 0.95
}
``` 