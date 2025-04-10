import nltk
import textstat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from typing import Dict, Any, List, Tuple
import logging
import spacy # Using spacy for tokenization might be more robust

from .config import PRONOUN_CATEGORIES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialization ---
try:
    # Ensure NLTK data is available (run once)
    # nltk.download('punkt', quiet=True)
    # nltk.download('averaged_perceptron_tagger', quiet=True) # Needed for some pronoun methods if not using spaCy
    pass # Assume downloaded if using regularly
except Exception as e:
    logging.warning(f"NLTK download check failed (might need manual download): {e}")

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Load spaCy model (more robust tokenization and POS tagging)
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # Disable unused components for speed
    logging.info("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None # Fallback or raise error? For now, allow fallback but warn heavily.
except Exception as e:
    logging.error(f"Error loading spaCy model: {e}")
    nlp = None

# --- Helper Functions ---
def get_sentiment_vader(text: str) -> Tuple[float, str]:
    """Calculates VADER sentiment compound score and assigns a label."""
    vs = vader_analyzer.polarity_scores(text)
    score = vs['compound']
    if score >= 0.05:
        label = 'positive'
    elif score <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return score, label

def analyze_pronoun_usage(doc) -> Dict[str, Dict[str, Any]]:
    """Analyzes pronoun usage using spaCy POS tagging."""
    pronoun_counts = Counter()
    total_words = len([token for token in doc if not token.is_punct and not token.is_space])

    # Flatten pronoun categories for easy lookup
    pronoun_map = {}
    for category, pronouns in PRONOUN_CATEGORIES.items():
        for pronoun in pronouns:
            pronoun_map[pronoun] = category

    for token in doc:
        # Use POS tagging for pronouns
        if token.pos_ == "PRON":
            # Use lemma for consistency (e.g., 'I', 'me' -> 'I' if lemmatizer active, but we use lower)
            # Using lower() is simpler here as categories are lowercase
            pronoun_lower = token.text.lower()
            pronoun_counts[pronoun_lower] += 1

    # Categorize counts
    category_counts = {cat: 0 for cat in PRONOUN_CATEGORIES}
    for pronoun, count in pronoun_counts.items():
        category = pronoun_map.get(pronoun)
        if category:
            category_counts[category] += count
        else:
            logging.debug(f"Uncategorized pronoun found: '{pronoun}'") # Log if needed

    # Calculate percentages and structure output
    results = {}
    for category, count in category_counts.items():
         percentage = (count / total_words * 100) if total_words > 0 else 0
         results[category] = {'count': count, 'percentage': round(percentage, 4)}

    return results

# --- Main Analysis Function ---
def calculate_metrics(text: str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Calculates various quantitative metrics for a given text entry.

    Args:
        text: The journal entry content.

    Returns:
        A tuple containing:
        - metrics: Dictionary of general text metrics.
        - pronoun_metrics: Dictionary of pronoun usage metrics by category.
    """
    metrics = {}
    pronoun_metrics = {cat: {'count': 0, 'percentage': 0.0} for cat in PRONOUN_CATEGORIES} # Default empty

    if not text or not isinstance(text, str):
        logging.warning("calculate_metrics received empty or invalid text.")
        return metrics, pronoun_metrics

    try:
        # Basic Text Stats
        metrics['word_count'] = textstat.lexicon_count(text, removepunct=True)
        metrics['sentence_count'] = textstat.sentence_count(text)
        metrics['avg_sentence_length'] = round(textstat.avg_sentence_length(text), 2) if metrics['sentence_count'] > 0 else 0

        # Readability
        try:
            metrics['reading_level_flesch'] = round(textstat.flesch_reading_ease(text), 2)
        except ZeroDivisionError:
             metrics['reading_level_flesch'] = 0.0 # Handle case with no sentences/words

        # Sentiment (VADER)
        sentiment_score, sentiment_label = get_sentiment_vader(text)
        metrics['sentiment_score_vader'] = round(sentiment_score, 4)
        metrics['sentiment_label_vader'] = sentiment_label

        # Pronoun Usage (using spaCy)
        if nlp:
            doc = nlp(text)
            pronoun_metrics = analyze_pronoun_usage(doc)
        else:
            logging.warning("spaCy model not loaded. Skipping pronoun analysis.")

    except Exception as e:
        logging.error(f"Error calculating metrics: {e}", exc_info=True)
        # Return partially filled metrics if possible, or empty ones
        # Ensure default keys exist even if calculation failed
        default_metrics = {
            'word_count': 0, 'sentence_count': 0, 'avg_sentence_length': 0.0,
            'reading_level_flesch': 0.0, 'sentiment_score_vader': 0.0,
            'sentiment_label_vader': 'neutral'
        }
        for key, default_val in default_metrics.items():
            if key not in metrics:
                metrics[key] = default_val

    return metrics, pronoun_metrics

# Example usage (for testing)
if __name__ == "__main__":
    sample_text = """
    9.22.14

    From a research paper, quoted from a woman in a long-term poly relationship: "I think the biggest commitment I have with Rogelio is to show up and tell the truth." This is not a commitment to make the relationship last, but rather, "It's a commitment to make the relationship as truthful, as mutually supportive, as evolving for the other person and for ourselves as possible."

    She goes on to say, "If you really want to be committed for the rest of your life to this person, you're going to need to keep showing up and finding out what's true in that moment so that you can keep evolving together, because if you're committed to some ideal or some pretense, it's going to backfire."

    These two statements relaxed me in an important way. I am still feeling a lot of internal pressure to construct and commit to a relationship model, not just a mode of interaction but an identity and a philosophy too. A few days ago I advised myself to philosophize less and love more. I'd expand that statement now to say, "Let your philosophy and identity grow out of your choices. You can love [FemaleName259] in your way. Your ideals don't have to match hers or anyone else's. Remain sensitive to what you want in this moment."
    """
    if nlp: # Only run example if spaCy loaded
        general_metrics, p_metrics = calculate_metrics(sample_text)
        print("--- General Metrics ---")
        for k, v in general_metrics.items():
            print(f"{k}: {v}")
        print("\n--- Pronoun Metrics ---")
        for cat, data in p_metrics.items():
            print(f"{cat}: Count={data['count']}, Percentage={data['percentage']:.2f}%")
    else:
        print("Cannot run example: spaCy model not loaded.") 