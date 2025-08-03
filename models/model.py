import torch
import torch.nn as nn


# Define a Multi-Layer Perceptron (MLP) model class for tabular data

class TabularMLP(nn.Module):

    def __init__(self, in_features, hidden_dims, dropout):
        """
            Initialize the MLP model.
            
            Args:
                in_features (int): Number of input features (dimension of input data).
                hidden_dims (list of int): Sizes of each hidden layer.
                dropout (float): Dropout rate applied after each hidden layer to reduce overfitting.
        """
        super().__init__()
        layers = []     # List to hold all layers sequentially
        last_dim = in_features      #Track the Previous size of the layer output

        # Build each hidden layer with Linear -> BatchNorm -> ReLU -> Dropout
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, h_dim),     # Fully connected linear layer
                nn.BatchNorm1d(h_dim),      # Batch normalization for faster and stable training
                nn.ReLU(),      # ReLU activation to introduce non-linearity
                nn.Dropout(dropout)     # Dropout to prevent overfitting
            ])
            last_dim = h_dim        # Update last_dim for next layer input size
        # Final output layer: single neuron for binary classification (output is a logit)
        layers.append(nn.Linear(last_dim, 1))
        # Combine all layers in sequential container
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
            Forward pass through the network.
            
            Args:
                x (Tensor): Input tensor of shape (batch_size, in_features)
            
            Returns:
                Tensor: Output logits tensor of shape (batch_size,), squeezed for convenience
        """
        return self.model(x).squeeze(1)        # Squeeze to remove extra dimension for output size (batch_size)

