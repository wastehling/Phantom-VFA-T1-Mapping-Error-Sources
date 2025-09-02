import torch

from models.NCDE.NCDE import NeuralCDE
from models.NN.NN import NN


def create_model(parameters, device):
    """
    creates or loads model object
    Args:
        parameters: parameters specifying model loading or setup
        device: cuda or cpu

    Returns:
        model: model object

    """
    if parameters['model']['pretrained']:
        model = torch.load(parameters['model']['pretrained_model_path'], map_location=device)

    else:
        # create new model
        if parameters['model']['family'] == 'NCDE':
            model = NeuralCDE(input_channels=parameters['model']['initial']['input_channels'],
                              parameters=parameters,
                              device=device).to(device)
        elif parameters['model']['family'] == 'NN':
            model = NN(parameters=parameters, device=device).to(device)
        else:
            raise NotImplementedError('Other models than NCDE or NN not implemented')

    return model


def load_model(path, device):
    """
    load model from path
    Args:
        path: path to load model from
        device: device to load model to

    Returns:
        model: model object

    """
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except:
        model = torch.load(path, map_location='cpu', weights_only=False)
        device = 'cpu'
    return model, device
