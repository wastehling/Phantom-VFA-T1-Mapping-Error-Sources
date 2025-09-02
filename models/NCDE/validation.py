import logging
import torch
import copy
import numpy as np

from tqdm import tqdm
from utils.data_modification.decompose_loss import decompose_loss_per_element, average_decomposed_losses
from qmri.forward_model.ivim import ivim_coefficients_to_parameters, ivim_parameters_to_signal
from qmri.forward_model.vfa import (vfa_coefficients_to_parameters, vfa_parameters_to_signal,
                                    vfa_parameters_to_signal_transformed)
from qmri.forward_model.dce import dce_coefficients_to_parameters, dce_parameters_to_signal


def validation_NCDE(model, dataloader, parameters, metric_list, device, criterion):
    """
    validation function of model based on data from val_loader
    Args:
        model: model object to validate
        parameters: parameters dict
        metric_list: list of metrics to use for validation
        dataloader: dataloader object containing data
        criterion: loss function
        device: either cuda or cpu

    Returns:
        signal_loss: loss between predicted signal and actual signal
        parameter_loss: loss between predicted parameters and actual parameters
        total_loss: total loss
        metric_list: list of metrics

    """
    steps = 0
    cum_signal_loss = 0
    cum_parameter_loss = 0
    cum_total_loss = 0

    model.eval()
    logging.info("computing validation loss")

    # initialize losses
    decomposed_signal_loss = {}
    decomposed_parameter_loss = {}

    # for data in val_data:
    for data, cubic_coeffs, parameter_coeffs, non_predicted_parameters, metadata in tqdm(dataloader):

        # Get the data from the batch
        x_vals = data[:, :, 0].float().to(device)
        signals = data[:, :, 1].float().to(device)
        observations = data[:, :, 2].float().to(device)
        observations_mask = torch.nn.functional.pad(observations[:, 1:] - observations[:, :-1], (1, 0), 'constant',
                                                    1).to(device)

        # parameter coeffs to device
        cubic_coeffs = cubic_coeffs.float().to(device)
        parameter_coeffs = parameter_coeffs.float().to(device)
        if non_predicted_parameters is not None:
            non_predicted_parameters = non_predicted_parameters.float().to(device)

        # data through model
        model_coefficients = model(cubic_coeffs, parameters['model']['CDE']['atol'],
                                   parameters['model']['CDE']['rtol'])

        if parameters['qmri_fit']['model'] == 'VFA':
            M0, TRoverT1 = vfa_coefficients_to_parameters(model_coefficients,
                                                          parameters['qmri_fit']['parameter_ranges'],
                                                          parameters['qmri_fit']['mapping_method'])
            TR = torch.reshape(non_predicted_parameters[:, 0], TRoverT1.size())
            T1 = TR / TRoverT1
            T1min, T1max = parameters['qmri_fit']['parameter_ranges']['T1'][0], \
            parameters['qmri_fit']['parameter_ranges']['T1'][1]
            M0min, M0max = parameters['qmri_fit']['parameter_ranges']['M0'][0], \
            parameters['qmri_fit']['parameter_ranges']['M0'][1]
            T1_linear_coeff = (T1 - T1min) / (T1max - T1min)
            M0_linear_coeff = (M0 - M0min) / (M0max - M0min)
            model_coefficients_linear = torch.cat((M0_linear_coeff, T1_linear_coeff), dim=1)
            parameter_criterion = torch.nn.MSELoss(reduction='none').to(device)
            parameter_loss_per_element = parameter_criterion(model_coefficients_linear, parameter_coeffs)
            parameter_coeffs_boolean = parameter_coeffs.bool().int()
            parameter_loss_per_element = parameter_coeffs_boolean * parameter_loss_per_element
            parameter_loss = torch.mean(parameter_loss_per_element)
            decomposed_parameter_loss = decompose_loss_per_element(decomposed_parameter_loss, parameter_loss_per_element, metadata)
        else:
            # compute parameter loss based on linear mapping
            model_coefficients_linear = torch.sigmoid(model_coefficients)
            parameter_criterion = torch.nn.MSELoss(reduction='none').to(device)
            parameter_loss_per_element = parameter_criterion(model_coefficients_linear, parameter_coeffs)
            parameter_coeffs_boolean = parameter_coeffs.bool().int()
            parameter_loss_per_element = parameter_coeffs_boolean * parameter_loss_per_element
            parameter_loss = torch.mean(parameter_loss_per_element)
            decomposed_parameter_loss = decompose_loss_per_element(decomposed_parameter_loss, parameter_loss_per_element, metadata)

        # signal prediction
        if parameters['qmri_fit']['model'] == 'IVIM':
            D, Dp, f, S0 = ivim_coefficients_to_parameters(model_coefficients,
                                                           parameters['qmri_fit']['parameter_ranges'],
                                                           parameters['qmri_fit']['mapping_method'])
            signal_pred = ivim_parameters_to_signal(D, Dp, f, S0, x_vals)

        elif parameters['qmri_fit']['model'] == 'VFA':
            M0, T1overTR = vfa_coefficients_to_parameters(model_coefficients,
                                                          parameters['qmri_fit']['parameter_ranges'],
                                                          parameters['qmri_fit']['mapping_method'])
            TRoverT1 = 1 / T1overTR
            TR = torch.reshape(non_predicted_parameters[:, 0], TRoverT1.size())
            T1 = TR / TRoverT1

            if 'transformed' in parameters['qmri_fit'].keys() and parameters['qmri_fit']['transformed']:
                _, signal_pred_transformed = vfa_parameters_to_signal_transformed(M0, T1, TR, x_vals)
                signal_pred = signal_pred_transformed * torch.sin(torch.deg2rad(x_vals))
                signals = signals * torch.sin(torch.deg2rad(x_vals))
            else:
                _, signal_pred = vfa_parameters_to_signal(M0, T1, TR, x_vals)

        elif parameters['qmri_fit']['model'] == 'DCE':
            ke, ve, vp, dt = dce_coefficients_to_parameters(model_coefficients,
                                                            parameters['qmri_fit']['parameter_ranges'],
                                                            parameters['qmri_fit']['mapping_method'])
            aif_dict = copy.deepcopy(parameters['qmri_fit']['aif']['aif'])
            aif_dict['ab'] /= (1 - parameters['qmri_fit']['aif']['Hct'])
            aif_arr = np.array([aif_dict['t0'], aif_dict['tr'], aif_dict['ab'], aif_dict['mb'], aif_dict['ae'],
                                aif_dict['me'], aif_dict['ar'], aif_dict['mm'], aif_dict['mr']])
            aif = torch.tensor(np.transpose(np.tile(aif_arr, (ve.size(0), 1))), dtype=torch.float, device=device)
            signal_pred = dce_parameters_to_signal(aif, ke, ve, vp, dt, x_vals, device)
        else:
            raise NotImplementedError('other models than IVIM, VFA and DCE not implemented')

        # calculate loss
        # correct for padding
        signals = torch.multiply(signals, observations_mask)
        signal_pred = torch.multiply(signal_pred, observations_mask)

        if parameters['training']['intensity_weighted']:
            signal_diff_per_element = torch.pow(torch.subtract(signal_pred, signals), 2)
            intensity_per_element = torch.mean(signals, dim=1)
            intensity_per_element = intensity_per_element.unsqueeze(-1).repeat(1, signal_diff_per_element.size(-1))
            signal_loss_per_element = torch.sum(signal_diff_per_element, dim=1) / torch.sum(observations_mask, dim=1)
            signal_loss = torch.mean(signal_loss_per_element)
            decomposed_signal_loss = decompose_loss_per_element(decomposed_signal_loss, signal_loss_per_element,
                                                                metadata)
            signal_diff_per_element.register_hook(lambda grad: grad * torch.nan_to_num(1 / \
                                                                            torch.pow(intensity_per_element, 2)))
        else:
            signal_diff_per_element = torch.pow(torch.subtract(signal_pred, signals), 2)
            signal_loss_per_element = torch.sum(signal_diff_per_element, dim=1) / torch.sum(observations_mask, dim=1)
            signal_loss = torch.mean(signal_loss_per_element)
            decomposed_signal_loss = decompose_loss_per_element(decomposed_signal_loss, signal_loss_per_element,
                                                                metadata)

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

    decomposed_parameter_loss = average_decomposed_losses(decomposed_parameter_loss)
    decomposed_signal_loss = average_decomposed_losses(decomposed_signal_loss)

    return signal_loss, parameter_loss, decomposed_parameter_loss, decomposed_signal_loss, total_loss, metric_list
