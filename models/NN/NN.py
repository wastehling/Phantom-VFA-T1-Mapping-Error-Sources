import torch

from models.NCDE.readout import Readout


class NN(torch.nn.Module):
    """
    class for constructing a neural network
    """
    def __init__(self, parameters={}, device=None):
        super(NN, self).__init__()

        assert parameters, "Parameters are required for NN"

        if parameters['model']['readout_type'] == 'Readout':
            self.readout = Readout(parameters, device)
        else:
            raise NotImplementedError("Only Readout readout type is implemented")

    def forward(self, signals):
        pred_y = self.readout(signals)
        return pred_y


