import torch
import numpy as np

def decompose_loss_per_element(decomposed_loss, loss_per_element, metadata):
    """
    Decompose loss per element based on metadata ([type, application, study])
    Args:
        loss_per_element: loss per element
        metadata: metadata per element

    Returns:
        decomposed_loss: decomposed loss
    """

    for idx, el in enumerate(metadata):
        type, application, study = el
        if study not in decomposed_loss.keys():
            decomposed_loss[study] = []
        decomposed_loss[study].append(loss_per_element[idx])
    return decomposed_loss


def average_decomposed_losses(decomposed_loss):
    """
    Average the decomposed loss per epoch
    Args:
        decomposed_loss: loss per study

    Returns:
        decomposed_loss: decomposed loss
    """

    for study in decomposed_loss.keys():
        decomposed_loss[study] = np.average(torch.stack(decomposed_loss[study]).detach().numpy(), axis=0)
    return decomposed_loss