import unittest
import torch
import torch.nn as nn
from model import NeuralNet

class TestNeuralNet(unittest.TestCase):
    """
    Testing strategy:
    1. Test initialization of the NeuralNet class.
    2. Test forward pass of the NeuralNet class.
    """

    def test_init(self):
        # Test initialization of the NeuralNet class
        input_size, hidden_size, num_classes = 10, 20, 5
        model = NeuralNet(input_size, hidden_size, num_classes)

        # Check if the model has the expected attributes
        self.assertIsInstance(model.l1, nn.Linear)
        self.assertIsInstance(model.l2, nn.Linear)
        self.assertIsInstance(model.l3, nn.Linear)
        self.assertIsInstance(model.relu, nn.ReLU)

        # Check if the sizes of linear layers match the specified sizes
        self.assertEqual(model.l1.in_features, input_size)
        self.assertEqual(model.l1.out_features, hidden_size)
        self.assertEqual(model.l2.in_features, hidden_size)
        self.assertEqual(model.l2.out_features, hidden_size)
        self.assertEqual(model.l3.in_features, hidden_size)
        self.assertEqual(model.l3.out_features, num_classes)

    def test_forward(self):
        # Test forward pass of the NeuralNet class
        input_size, hidden_size, num_classes = 10, 20, 5
        model = NeuralNet(input_size, hidden_size, num_classes)

        # Create a sample input tensor
        x = torch.randn(1, input_size)

        # Perform the forward pass
        output = model.forward(x)

        # Check if the output tensor has the correct shape
        self.assertEqual(output.shape, torch.Size([1, num_classes]))

if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestNeuralNet)

    # Run the test suite
    # runner = unittest.TextTestRunner()
    # result = runner.run(suite)

    # # Print the test results
    # print(result)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)

    # Print the test results
    print("\n=== Test Results ===")
    print(f"Number of tests run: {test_result.testsRun}")
    print(f"Number of failures: {len(test_result.failures)}")
    print(f"Number of errors: {len(test_result.errors)}")
    print(f"Number of skipped tests: {len(test_result.skipped)}")
    print(f"Was successful?: {test_result.wasSuccessful()}")
