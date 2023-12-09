import unittest
from unittest.mock import patch
from chat import get_response  # Replace 'your_script_name' with the name of your script containing the get_response function

class TestGetResponse(unittest.TestCase):
    # Testing strategy:
    # - Empty input
    # - Typical sentences/questions
    # - Sentences with unknown words
    # - Quit command input ("quit")
    # - Margin input cases close to the threshold probability (around 0.75)
    # - Test cases covering various intents from the `intents.json` file

    @patch('builtins.input', side_effect=["", "Hello!", "How are you?", "quit"])
    def test_empty_and_typical_input(self, mock_input):
        # Test empty input
        self.assertEqual(get_response(""), "I do not understand...")

        # Test typical sentences
        responses = [get_response("Hello!"), get_response("How are you?")]
        self.assertNotIn("I do not understand...", responses)

    def test_unknown_words(self):
        # Test input with unknown words
        response = get_response("Some random unknown words")
        self.assertEqual(response, "I do not understand...")

    # def test_quit_command(self):
    #     # Test 'quit' command
    #     with self.assertRaises(SystemExit):
    #         get_response("quit")

if __name__ == '__main__':
    # Run the tests and print the results
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestGetResponse)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)

    # Print the test results
    print("\n=== Test Results ===")
    print(f"Number of tests run: {test_result.testsRun}")
    print(f"Number of failures: {len(test_result.failures)}")
    print(f"Number of errors: {len(test_result.errors)}")
    print(f"Number of skipped tests: {len(test_result.skipped)}")
    print(f"Was successful?: {test_result.wasSuccessful()}")
