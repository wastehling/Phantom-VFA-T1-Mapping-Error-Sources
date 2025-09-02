import torch
import logging
import numpy as np
import copy

from tqdm import tqdm
from qmri.forward_model.ivim import ivim_coefficients_to_parameters, ivim_parameters_to_signal
from qmri.forward_model.vfa import vfa_coefficients_to_parameters, vfa_parameters_to_signal
from qmri.forward_model.dce import dce_coefficients_to_parameters, dce_parameters_to_signal


def validation_NN(model, dataloader, parameters, metric_list, device, criterion):
    """
    train function of model based on data from train_loader, val_loader and sim_loader
    Args:
        metric_list:
        model: model object to train
        parameters: parameters dict
        metric_list: list of metrics to use for validation
        dataloader: dataloader object containing data
        criterion: loss function
        device: either cuda or cpu

    Returns:
        signal_loss: loss between predicted signal and actual signal
        parameter_loss: loss between predicted parameters and actual parameters

    """
    steps = 0
    cum_signal_loss = 0
    cum_parameter_loss = 0
    cum_total_loss = 0

    model.eval()
    logging.info("computing validation loss")

    # for data in val_data:
    for data, cubic_coeffs, parameter_coeffs, non_predicted_parameters in tqdm(dataloader):

        # Get the data from the batch
        xvals = data[:, :, 0].float().to(device)
        signals = data[:, :, 1].float().to(device)

        # parameter coeffs to device
        parameter_coeffs = parameter_coeffs.float().to(device)
        if non_predicted_parameters is not None:
            non_predicted_parameters = non_predicted_parameters.float().to(device)

        # data through model
        model_coefficients = model(signals)

        if parameters['training']['parameter_loss_fraction'] > 0:
            parameter_loss = criterion(model_coefficients, parameter_coeffs)
        else:
            parameter_loss = torch.tensor(0).to(device)

        # signal prediction
        if parameters['qmri_fit']['model'] == 'IVIM':
            D, Dp, f, S0 = ivim_coefficients_to_parameters(model_coefficients,
                                                           parameters['qmri_fit']['parameter_ranges'],
                                                           parameters['qmri_fit']['mapping_method'])
            signal_pred = ivim_parameters_to_signal(D, Dp, f, S0, xvals)

        elif parameters['qmri_fit']['model'] == 'VFA':
            M0, TRoverT1 = vfa_coefficients_to_parameters(model_coefficients,
                                                    parameters['qmri_fit']['parameter_ranges'],
                                                    parameters['qmri_fit']['mapping_method'])
            TR = torch.reshape(non_predicted_parameters[:, 0], TRoverT1.size())
            T1 = TR / TRoverT1
            _, signal_pred = vfa_parameters_to_signal(M0, T1, TR, xvals)
        elif parameters['qmri_fit']['model'] == 'DCE':
            ke, ve, vp, dt = dce_coefficients_to_parameters(model_coefficients,
                                                            parameters['qmri_fit']['parameter_ranges'],
                                                            parameters['qmri_fit']['mapping_method'])
            aif_dict = copy.deepcopy(parameters['qmri_fit']['aif']['aif'])
            aif_dict['ab'] /= (1 - parameters['qmri_fit']['aif']['Hct'])
            aif_arr = np.array([aif_dict['t0'], aif_dict['tr'], aif_dict['ab'], aif_dict['mb'], aif_dict['ae'],
                                aif_dict['me'], aif_dict['ar'], aif_dict['mm'], aif_dict['mr']])
            aif = torch.tensor(np.transpose(np.tile(aif_arr, (ve.size(0), 1))), dtype=torch.float, device=device)
            signal_pred = dce_parameters_to_signal(aif, ke, ve, vp, dt, xvals, device)
        else:
            raise NotImplementedError('other models than IVIM and VFA not yet implemented')

        # calculate signal loss
        signal_loss = criterion(signal_pred, signals)

        # compute total loss
        total_loss = (1 - parameters['training']['parameter_loss_fraction']) * signal_loss + \
                     parameters['training']['parameter_loss_fraction'] * parameter_loss

        cum_signal_loss = cum_signal_loss + signal_loss.cpu().detach().numpy()
        cum_parameter_loss = cum_parameter_loss + parameter_loss.cpu().detach().numpy()
        cum_total_loss = cum_total_loss + total_loss.cpu().detach().numpy()
        steps += 1

        # compute metrics
        for metric in metric_list:
            metric.compute(torch.flatten(signal_pred).cpu().detach().numpy(),
                           torch.flatten(signals).cpu().detach().numpy())
            metric.update()

    signal_loss = cum_signal_loss / steps
    parameter_loss = cum_parameter_loss / steps
    total_loss = cum_total_loss / steps

    for metric in metric_list:
        metric.average()
        metric.log('signal')
        metric.clear_cache()

    return signal_loss, parameter_loss, total_loss, metric_list

