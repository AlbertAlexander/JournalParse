import re
from typing import Dict, Any

def analyze_pronouns(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Analyze pronoun usage in text.
    Returns dict with counts and percentages for different pronoun categories.
    """
    pronouns = {
        'first_person_singular': ['i', 'me', 'my', 'mine', 'myself'],
        'first_person_plural': ['we', 'us', 'our', 'ours', 'ourselves'],
        'second_person': ['you', 'your', 'yours', 'yourself', 'yourselves'],
        'third_person': ['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
                        'herself', 'it', 'its', 'itself', 'they', 'them', 
                        'their', 'theirs', 'themselves']
    }
    
    # Convert to lowercase for matching
    text = text.lower()
    
    # Count pronouns
    results = {}
    total_pronouns = 0
    
    for category, pronoun_list in pronouns.items():
        count = sum(len(re.findall(r'\b' + p + r'\b', text)) for p in pronoun_list)
        total_pronouns += count
        results[category] = {'count': count, 'percentage': 0.0}
    
    # Calculate percentages
    if total_pronouns > 0:
        for category in results:
            results[category]['percentage'] = (results[category]['count'] / total_pronouns) * 100
    
    return results 