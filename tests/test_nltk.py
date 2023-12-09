import unittest
import numpy as np
from nltk_utils import tokenize, stem, bag_of_words 

class TestYourFunctions(unittest.TestCase):
    # Testing strategy:
    # For tokenize function:
    # - Empty string input
    # - Sentence with punctuation
    # - Sentence with numbers
    # - Typical sentence with words

    def test_tokenize_empty_string(self):
        sentence = ""
        expected_tokens = []
        self.assertEqual(tokenize(sentence), expected_tokens)

    def test_tokenize_with_punctuation(self):
        sentence = "This is a test, with punctuation."
        expected_tokens = ['This', 'is', 'a', 'test', ',', 'with', 'punctuation', '.']
        self.assertEqual(tokenize(sentence), expected_tokens)

    def test_tokenize_with_numbers(self):
        sentence = "There are 123 apples."
        expected_tokens = ['There', 'are', '123', 'apples', '.']
        self.assertEqual(tokenize(sentence), expected_tokens)

    def test_tokenize_typical_sentence(self):
        sentence = "The quick brown fox jumps over the lazy dog."
        expected_tokens = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog', '.']
        self.assertEqual(tokenize(sentence), expected_tokens)

    # For stem function:
    # - Empty string input
    # - Words with uppercase letters
    # - Typical words to stem

    def test_stem_empty_string(self):
        word = ""
        self.assertEqual(stem(word), '')

    def test_stem_with_uppercase(self):
        word = "Organize"
        expected_stem = "organ"
        self.assertEqual(stem(word), expected_stem)

    def test_stem_typical_words(self):
        words = ["organize", "organizes", "organizing"]
        expected_stems = ["organ", "organ", "organ"]
        stemmed_words = [stem(w) for w in words]
        self.assertEqual(stemmed_words, expected_stems)

    # For bag_of_words function:
    # - Empty tokenized sentence
    # - Empty words list
    # - Typical sentence and words list

    def test_bag_of_words_empty_sentence(self):
        sentence = []
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        expected_bag = np.zeros(len(words), dtype=np.float32)
        self.assertTrue(np.array_equal(bag_of_words(sentence, words), expected_bag))

    def test_bag_of_words_empty_words(self):
        sentence = ["hello", "how", "are", "you"]
        words = []
        expected_bag = np.zeros(0, dtype=np.float32)
        self.assertTrue(np.array_equal(bag_of_words(sentence, words), expected_bag))

    def test_bag_of_words_typical_case(self):
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        expected_bag = np.array([0, 1, 0, 1, 0, 0, 0], dtype=np.float32)
        self.assertTrue(np.array_equal(bag_of_words(sentence, words), expected_bag))

if __name__ == '__main__':
    # Run the tests and print the results
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestYourFunctions)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)

    # Print the test results
    print("\n=== Test Results ===")
    print(f"Number of tests run: {test_result.testsRun}")
    print(f"Number of failures: {len(test_result.failures)}")
    print(f"Number of errors: {len(test_result.errors)}")
    print(f"Number of skipped tests: {len(test_result.skipped)}")
    print(f"Was successful?: {test_result.wasSuccessful()}")
