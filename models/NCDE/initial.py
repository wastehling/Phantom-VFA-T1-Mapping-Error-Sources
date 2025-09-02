import torch


class Initial(torch.nn.Module):
    """
    Class dedicated to creating a neural network transforming a dimensionality, input_channels, to a different
    dimensionality, hidden_channels
    """
    def __init__(self, parameters):
        super(Initial, self).__init__()
        input_channels = parameters['model']['initial']['input_channels']
        hidden_channels = parameters['model']['initial']['hidden_channels']
        self.initial = torch.nn.Linear(input_channels, hidden_channels)

    def forward(self, x):
        hidden_state = self.initial(x)
        return hidden_state


