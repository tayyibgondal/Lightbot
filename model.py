import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    """
    Neural Network model for intent classification.

    Attributes:
    - l1 (nn.Linear): First linear layer.
    - l2 (nn.Linear): Second linear layer.
    - l3 (nn.Linear): Third linear layer.
    - relu (nn.ReLU): Rectified Linear Unit activation function.

    Methods:
    - __init__(self, input_size, hidden_size, num_classes): Constructor to initialize the model.
    - forward(self, x): Forward pass of the model.
    """

    def __init__(self, input_size, hidden_size, num_classes):
        """
        Constructor to initialize the neural network model.

        Parameters:
        - input_size (int): Size of the input features.
        - hidden_size (int): Size of the hidden layer.
        - num_classes (int): Number of output classes.
        """
        super(NeuralNet, self).__init__()
        
        # Define linear layers and activation function
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor.
        """
        # Forward pass through linear and activation layers
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)


        # No activation and no softmax at the end (intended for use with CrossEntropyLoss)
        return out
