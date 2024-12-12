import torch.nn as nn


class MyModel(nn.Module):
    """
    Class for training a model.

    Attributes:
        model: PyTorch model.

    """

    def __init__(self, inp, out):
        """
        The constructor for MyModel class.

        Parameters:
            inp (int): length of data vector that input to model.
            out (int): length of data vector that output from model.
        """
        super().__init__()
        self.model = nn.Sequential(nn.Linear(inp, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, out))

    def forward(self, data):
        """
        The function that applies model layers to the data.

        Parameters:
            data (tensor): train data.

        Returns:
            tensor: model prediction in the form of a probability vector.
        """
        return self.model(data)
