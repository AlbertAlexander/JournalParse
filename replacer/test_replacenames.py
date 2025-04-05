import unittest
from replacenames import replace_terms

class TestReplacenames(unittest.TestCase):
    def setUp(self):
        # Test mapping dictionary with exact case preservation
        self.mappings = {
            "names": {
                "Tom": "[Name1]",
                "Mary Jane": "[Name2]",
                "O'Brien": "[Name3]",
                "ACME": "[BUSINESS1]",
                "DeadLetter": "[Business2]"
            }
        }

    def test_case_sensitive_matching(self):
        """Test that matches are case sensitive"""
        text = "tom Tom TOM Tommy Tomcat"
        result, counts = replace_terms(text, self.mappings)
        self.assertEqual(result, "tom [Name1] TOM Tommy Tomcat")
        self.assertEqual(counts[("Tom", "[Name1]", "names")], 1)

    def test_possessives(self):
        """Test possessive forms.
        Note: Only singular possessives are replaced. Plural possessives are preserved
        to avoid false positives with actual plurals.
        """
        text = (
            "Tom's house, "     # Singular possessive - should replace
            "Toms' houses, "    # Plural possessive - should NOT replace
            "Tom house, "       # Regular form - should replace
            "Toms houses"       # Plural form - should NOT replace
        )
        result, counts = replace_terms(text, self.mappings)
        self.assertEqual(
            result,
            "[Name1]'s house, "  # Replaced
            "Toms' houses, "     # Preserved
            "[Name1] house, "    # Replaced
            "Toms houses"        # Preserved
        )
        self.assertEqual(counts[("Tom", "[Name1]", "names")], 2)  # Only 2 replacements

    def test_punctuation_boundaries(self):
        """Test word boundaries with punctuation"""
        text = "Tom! Tom, Tom. Tom? Tom; Tom\"Tom\""
        result, counts = replace_terms(text, self.mappings)
        self.assertEqual(result, "[Name1]! [Name1], [Name1]. [Name1]? [Name1]; [Name1]\"[Name1]\"")
        self.assertEqual(counts[("Tom", "[Name1]", "names")], 7)

    def test_compound_names(self):
        """Test multi-word names and special characters"""
        text = "Mary Jane's book. O'Brien's pub. mary jane MARY JANE"
        result, counts = replace_terms(text, self.mappings)
        self.assertEqual(result, "[Name2]'s book. [Name3]'s pub. mary jane MARY JANE")
        self.assertEqual(counts[("Mary Jane", "[Name2]", "names")], 1)

    def test_acronyms(self):
        """Test acronym handling"""
        text = "ACME Corp, Acme corp, acme corp"
        result, counts = replace_terms(text, self.mappings)
        self.assertEqual(result, "[BUSINESS1] Corp, Acme corp, acme corp")
        self.assertEqual(counts[("ACME", "[BUSINESS1]", "names")], 1)

    def test_case_preservation(self):
        """Test that replacement preserves the case of the matched term"""
        self.mappings["names"].update({
            "IBM": "[BUSINESS3]",
            "Microsoft": "[Business4]"
        })
        text = "IBM, Microsoft, microsoft, MICROSOFT"
        result, counts = replace_terms(text, self.mappings)
        self.assertEqual(result, "[BUSINESS3], [Business4], microsoft, MICROSOFT")
        self.assertEqual(counts[("IBM", "[BUSINESS3]", "names")], 1)
        self.assertEqual(counts[("Microsoft", "[Business4]", "names")], 1)

    def test_partial_matches(self):
        """Test that partial matches are not replaced and case is preserved"""
        text = "DeadLetter DEADLETTER deadletter DeadLetters PreDeadLetter LetterDead"
        result, counts = replace_terms(text, self.mappings)
        self.assertEqual(result, "[Business2] DEADLETTER deadletter DeadLetters PreDeadLetter LetterDead")
        self.assertEqual(counts[("DeadLetter", "[Business2]", "names")], 1)

    def test_word_boundaries(self):
        """Test word boundaries with case preservation"""
        self.mappings["names"].update({
            "Cat": "[Animal1]",
            "CAT": "[ANIMAL1]",
            "DeadLetter": "[Business2]"
        })
        text = (
            "Cat CAT cat Cats CatFood "          # Case variations and suffix
            "PreCat BlackCat "                   # Prefix tests
            "CatInTheHat TheCat "               # Compound words
            "Cat's CAT's cat's "                # Case-sensitive possessives
            "DEADLETTER DeadLetter deadletter"  # More case variations
        )
        result, counts = replace_terms(text, self.mappings)
        self.assertEqual(
            result,
            "[Animal1] [ANIMAL1] cat Cats CatFood "
            "PreCat BlackCat "
            "CatInTheHat TheCat "
            "[Animal1]'s [ANIMAL1]'s cat's "
            "DEADLETTER [Business2] deadletter"
        )
        self.assertEqual(counts[("Cat", "[Animal1]", "names")], 2)  # Matches "Cat" and "Cat's"
        self.assertEqual(counts[("CAT", "[ANIMAL1]", "names")], 2)  # Matches "CAT" and "CAT's"
        self.assertEqual(counts[("DeadLetter", "[Business2]", "names")], 1)  # Only exact case match

if __name__ == '__main__':
    unittest.main() 