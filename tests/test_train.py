import unittest
from unittest.mock import patch, mock_open
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from train import load_intents, preprocess_intents, create_training_data, create_dataset, train_model, save_model, NeuralNet

class TestTrainingMethods(unittest.TestCase):
    """
    Testing strategy:
    1. Test loading intents from a JSON file.
    2. Test preprocessing intents data.
    3. Test creating training data.
    4. Test creating a PyTorch dataset from training data.
    5. Test training the neural network model.
    6. Test saving the trained model to a file.
    """

    def test_load_intents(self):
        # Test loading intents from a JSON file
        with patch("builtins.open", mock_open(read_data='{"intents": []}')):
            intents = load_intents('intents.json')
        self.assertIsInstance(intents, dict, "Loading intents failed")

    def test_preprocess_intents(self):
        # Test preprocessing intents data
        intents = {'intents': [{'tag': 'greeting', 'patterns': ['Hi', 'Hello']}, {'tag': 'goodbye', 'patterns': ['Bye']}]}
        all_words, tags, xy = preprocess_intents(intents)
        self.assertIsInstance(all_words, list, "Preprocessing intents failed")
        self.assertIsInstance(tags, list, "Preprocessing intents failed")
        self.assertIsInstance(xy, list, "Preprocessing intents failed")

    def test_create_training_data(self):
        # Test creating training data
        all_words = ['hi', 'hello', 'bye']
        tags = ['greeting', 'goodbye']
        xy = [(['hi'], 'greeting'), (['hello'], 'greeting'), (['bye'], 'goodbye')]
        X_train, y_train = create_training_data(all_words, tags, xy)
        self.assertIsInstance(X_train, np.ndarray, "Creating training data failed")
        self.assertIsInstance(y_train, np.ndarray, "Creating training data failed")

    def test_create_dataset(self):
        # Test creating a PyTorch dataset from training data
        X_train = np.array([[1, 0, 1], [0, 1, 0]])
        y_train = np.array([0, 1])
        dataset = create_dataset(X_train, y_train)
        self.assertIsInstance(dataset, Dataset, "Creating dataset failed")

    def test_train_model(self):
        # Test training the neural network model
        input_size, hidden_size, output_size = 3, 4, 2
        model = NeuralNet(input_size, hidden_size, output_size)
        arr1 = np.array([1, 0, 1], dtype=np.float32)
        arr2 = np.array([0, 1, 0], dtype=np.float32)
        arr3 = np.array([0, 1], dtype=np.float32)
        train_loader = DataLoader(dataset=create_dataset(np.array([arr1, arr2]), arr3),
                                  batch_size=1, shuffle=True, num_workers=0)
        trained_model, final_loss = train_model(model, train_loader, num_epochs=2)
        self.assertIsInstance(trained_model, NeuralNet, "Training model failed")
        self.assertIsInstance(final_loss, float, "Training model failed")

    def test_save_model(self):
        # Test saving the trained model to a file
        input_size, hidden_size, output_size = 3, 4, 2
        model = NeuralNet(input_size, hidden_size, output_size)
        all_words, tags = ['hi', 'hello', 'bye'], ['greeting', 'goodbye']
        with patch('torch.save') as mock_save:
            save_model(model, input_size, hidden_size, output_size, all_words, tags)
            mock_save.assert_called_once()

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestTrainingMethods)
    
    # Run the test suite
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)

    # Print the test results
    print("\n=== Test Results ===")
    print(f"Number of tests run: {test_result.testsRun}")
    print(f"Number of failures: {len(test_result.failures)}")
    print(f"Number of errors: {len(test_result.errors)}")
    print(f"Number of skipped tests: {len(test_result.skipped)}")
    print(f"Was successful?: {test_result.wasSuccessful()}")