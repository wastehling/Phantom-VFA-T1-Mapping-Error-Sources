import torch
import numpy as np
import torchio
import copy

from tqdm import tqdm
from torch.utils.data import DataLoader
from qmri.forward_model.ivim import ivim_coefficients_to_parameters, ivim_parameters_to_signal
from qmri.forward_model.vfa import vfa_coefficients_to_parameters, vfa_parameters_to_signal, vfa_parameters_to_signal_transformed
from qmri.forward_model.dce import dce_coefficients_to_parameters, dce_parameters_to_signal
from utils.data_modification.modify_data import multiple_lists_to_list_of_dicts
from utils.data_modification.collate_data import DataCollater


def prediction(model, subject, parameters, device):
    """
    evaluates a model on in vivo data from dataloader
    Args:
        model: model to use for evaluation
        parameters: parameters
        subject: subject containing test data
        device: device, either cuda or cpu

    Returns:
        predicted_parameters_list_flattened: list of predicted parameters
        predicted_signals_list: list of predicted signals
    """

    # get data from subject
    signals_array = subject.get('signals').numpy()
    xvals_array = subject.get('xvals').numpy()
    if parameters['qmri_fit']['model'] == 'VFA':
        non_predicted_parameters_array = subject.get('TR_image').numpy()
    elif parameters['qmri_fit']['model'] == 'DCE':
        non_predicted_parameters_array = subject.get('AIF_image').numpy()
    else:
        non_predicted_parameters = None
        non_predicted_parameters_array = None

    # reshape data
    signals = np.reshape(signals_array, (signals_array.shape[0], signals_array.shape[1] *
                                         signals_array.shape[2] *
                                         signals_array.shape[3])).transpose().tolist()
    xvals = np.reshape(xvals_array, (xvals_array.shape[0], xvals_array.shape[1] *
                                     xvals_array.shape[2] *
                                     xvals_array.shape[3])).transpose().tolist()
    xvals *= len(signals)

    if non_predicted_parameters_array is not None:
        non_predicted_parameters = np.reshape(non_predicted_parameters_array,
                                              (non_predicted_parameters_array.shape[0],
                                                        non_predicted_parameters_array.shape[1] *
                                                        non_predicted_parameters_array.shape[2] *
                                                        non_predicted_parameters_array.shape[3])).transpose().tolist()

    # get valid mask
    try:
        mask_array = subject.get('valid_mask').numpy().flatten()
    except AttributeError:
        mask_array = np.ones(signals_array.shape[1:]).flatten()

    # retrieve valid signals and store valid indices
    valid_idcs = np.where(mask_array)[0]
    valid_signals = np.array(signals)[mask_array.astype(bool)].tolist()
    xvals *= len(valid_signals)

    # flatten data
    flattened_data = multiple_lists_to_list_of_dicts(valid_signals, 'signals', xvals, 'xvals',
                                                     non_predicted_parameters, 'non_predicted_parameters')

    # create data loader
    test_loader = DataLoader(flattened_data, batch_size=parameters['test']['batch_size'],
                             shuffle=False, collate_fn=DataCollater(parameters['qmri_fit']['model']).collate_data, drop_last=False)

    # predict parameters
    predicted_parameters_list, predicted_signals_list = [], []
    for data, coeffs, non_predicted_parameters in tqdm(test_loader):

        # predict parameters for one batch
        predicted_parameters, signals, predicted_signals = prediction_step_NN(model, data, parameters,
                                                                              non_predicted_parameters, device)
        predicted_parameters_list.append(predicted_parameters)

    # add predicted parameters to subject
    for key in predicted_parameters_list[0].keys():
        valid_array = np.concatenate([sublist[key] for sublist in predicted_parameters_list], axis=0).squeeze()
        if len(valid_array.shape) == 2:
            first_dim = valid_array.shape[1]
        else:
            first_dim = 1
        array = np.ones(
            (signals_array.shape[1] * signals_array.shape[2] * signals_array.shape[3], first_dim)).squeeze()
        array[:] = np.nan
        array[valid_idcs] = valid_array
        structured_array = np.reshape(np.transpose(array), (first_dim, signals_array.shape[1],
                                                            signals_array.shape[2], signals_array.shape[3]))
        subject.add_image(torchio.Image(tensor=torch.Tensor(structured_array)), f'prediction_{key}')

    # add computed residuals to subject
    residuals = np.power(subject.get('signals').numpy() - subject.get('prediction_signal_pred').numpy(), 2)
    residuals = np.expand_dims(np.sum(residuals, 0) * 1000, 0)
    subject.add_image(torchio.Image(tensor=torch.Tensor(residuals)), 'residuals')

    return subject


def prediction_step_NN(model, data, parameters, non_predicted_parameters, device):
    """
    step of prediction from data using model
    Args:
        model: model to use for evaluation
        parameters: parameters
        non_predicted_parameters: parameters that are used in prediction, but are known a priori (such as MR acquisition settings)
        data: data
        device: device, either cuda or cpu

    Returns:
        prediction: predicted parameters
        signals: supervision signals
        signal_pred: predicted signals
    """
    # initialize prediction dict
    prediction = {}

    # Get the data from the batch
    xvals = data[:, :, 0].float().to(device)
    signals = data[:, :, 1].float().to(device)

    # data through model
    model_coefficients = model(signals)

    if parameters['qmri_fit']['model'] == 'IVIM':
        D, Dp, f, S0 = ivim_coefficients_to_parameters(model_coefficients, parameters['qmri_fit']['parameter_ranges'],
                                                       parameters['qmri_fit']['mapping_method'])
        signal_pred = ivim_parameters_to_signal(D, Dp, f, S0, xvals)
        prediction['signal_pred'] = signal_pred.cpu().detach().numpy()
        prediction['D'] = D.cpu().detach().numpy()
        prediction['Dp'] = Dp.cpu().detach().numpy()
        prediction['f'] = f.cpu().detach().numpy()
        prediction['S0'] = S0.cpu().detach().numpy()
    elif parameters['qmri_fit']['model'] == 'VFA':
        M0, T1overTR = vfa_coefficients_to_parameters(model_coefficients, parameters['qmri_fit']['parameter_ranges'],
                                                parameters['qmri_fit']['mapping_method'])
        TRoverT1 = 1 / T1overTR
        TR = torch.reshape(torch.tensor(non_predicted_parameters, dtype=torch.float, device=device), (M0.size()))
        T1 = TR / TRoverT1
        if 'transformed' in parameters['qmri_fit'].keys() and parameters['qmri_fit']['transformed']:
            _, signal_pred_transformed = vfa_parameters_to_signal_transformed(M0, T1, TR, xvals)
            signal_pred = signal_pred_transformed * torch.sin(torch.deg2rad(xvals))
            signals = signals * torch.sin(torch.deg2rad(xvals))
        else:
            _, signal_pred = vfa_parameters_to_signal(M0, T1, TR, xvals)

        prediction['signal_pred'] = signal_pred.cpu().detach().numpy()
        prediction['M0'] = M0.cpu().detach().numpy()
        prediction['T1'] = T1.cpu().detach().numpy()
        prediction['TR'] = TR.cpu().detach().numpy()
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
        prediction['signal_pred'] = signal_pred.cpu().detach().numpy()
        prediction['ke'] = ke.cpu().detach().numpy()
        prediction['ve'] = ve.cpu().detach().numpy()
        prediction['vp'] = vp.cpu().detach().numpy()
        prediction['dt'] = dt.cpu().detach().numpy()
    else:
        raise NotImplementedError('other models than IVIM, VFA and DCE not yet implemented')

    return prediction, signals, signal_pred
