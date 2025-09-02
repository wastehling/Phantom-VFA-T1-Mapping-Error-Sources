import torch
import numpy as np
import torchio
import time

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.data_modification.modify_data import multiple_lists_to_list_of_dicts
from utils.data_modification.collate_data import DataCollater

from models.classical_fitting.LSQ.least_squares import fit_least_squares_array_ivim, fit_least_squares_array_vfa, \
     fit_least_squares_array_t2star
from qmri.forward_model.ivim import ivim_parameters_to_signal
from qmri.forward_model.vfa import vfa_parameters_to_signal
from qmri.forward_model.dce import fit_tofts_model, Cosine8AIF_ExtKety
from qmri.forward_model.T2star import T2star_parameters_to_signal


def prediction(model, subject, parameters, device):
    """
    evaluates a torchio subject using a predefined model with a predict function. The API of this function is equal to
    the other prediction functions of the models in the models folder.
    Args:
        model: model to use for evaluation, LSQ
        parameters: parameters
        subject: subject containing test data
        device: device, either cuda or cpu

    Returns:
        predicted_parameters_list_flattened: list of predicted parameters
        predicted_signals_list: list of predicted signals
    """

    # load subject data
    signals_array = subject.get('signals').numpy()
    xvals_array = subject.get('xvals').numpy()
    if parameters['qmri_fit']['model'] == 'IVIM':
        non_predicted_parameters_array = None
        non_predicted_parameters = None
    elif parameters['qmri_fit']['model'] == 'VFA':
        non_predicted_parameters_array = subject.get('TR_image').numpy()
    elif parameters['qmri_fit']['model'] == 'DCE':
        non_predicted_parameters_array = subject.get('AIF_image').numpy()
    elif parameters['qmri_fit']['model'] == 'T2star':
        non_predicted_parameters_array = None
        non_predicted_parameters = None

    # reshape signals_array and xvals_array
    signals = np.reshape(signals_array, (signals_array.shape[0], signals_array.shape[1] *
                                         signals_array.shape[2] *
                                         signals_array.shape[3])).transpose().tolist()
    xvals = np.reshape(xvals_array, (xvals_array.shape[0], xvals_array.shape[1] *
                                     xvals_array.shape[2] *
                                     xvals_array.shape[3])).transpose().tolist()

    # reshape non_predicted_parameters_array
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

    # retrieve valid signals and store their indices
    valid_idcs = np.where(mask_array)[0]
    valid_signals = np.array(signals)[mask_array.astype(bool)].tolist()

    if non_predicted_parameters is not None:
        non_predicted_parameters = np.array(non_predicted_parameters)[mask_array.astype(bool)].tolist()

    xvals *= len(valid_signals)

    # flatten data
    flattened_data = multiple_lists_to_list_of_dicts(valid_signals, 'signals', xvals, 'xvals',
                                                     non_predicted_parameters, 'non_predicted_parameters')

    # create data loader
    test_loader = DataLoader(flattened_data, batch_size=parameters['test']['batch_size'],
                             shuffle=False, collate_fn=DataCollater(parameters['qmri_fit']['model']).collate_data,
                             drop_last=False)

    # predict parameters and signals
    predicted_parameters_list, predicted_signals_list = [], []
    for data, coeffs, non_predicted_parameters in tqdm(test_loader):
        predicted_parameters, signals, predicted_signals = prediction_step_LSQ(model, data, parameters,
                                                                               non_predicted_parameters, device)
        predicted_parameters_list.append(predicted_parameters)

    # recreate predicted parameters array and store predicted parameters and signals in subject
    for key in predicted_parameters_list[0].keys():

        # reconstruct array of predicted parameters
        valid_array = np.concatenate([sublist[key] for sublist in predicted_parameters_list], axis=0).squeeze()
        if len(valid_array.shape) == 2:
            first_dim = valid_array.shape[1]
        else:
            first_dim = 1

        # create array with nans and fill with predicted parameters at valid indices
        array = np.ones((signals_array.shape[1] * signals_array.shape[2] * signals_array.shape[3], first_dim)).squeeze()
        array[:] = np.nan
        array[valid_idcs] = valid_array
        structured_array = np.reshape(np.transpose(array), (first_dim, signals_array.shape[1],
                                                            signals_array.shape[2], signals_array.shape[3]))
        subject.add_image(torchio.Image(tensor=torch.Tensor(structured_array)), f'prediction_{key}')

    # calculate residuals and store in subject
    residuals = np.power(subject.get('signals').numpy() - subject.get('prediction_signal_pred').numpy(), 2)
    residuals = np.expand_dims(np.sum(residuals, 0) * 1000, 0)
    subject.add_image(torchio.Image(tensor=torch.Tensor(residuals)), 'residuals')

    return subject


def prediction_step_LSQ(model, data, parameters, non_predicted_parameters, device):
    """
    step of prediction from data using LSQ
    Args:
        model: model to use for evaluation
        non_predicted_parameters: parameters that are used in prediction, but are known a priori (such as MR acquisition settings)
        parameters: parameters
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
    xvals = data[:, :, 0].numpy()
    signals = data[:, :, 1].numpy()

    if parameters['qmri_fit']['model'] == 'IVIM':
        # set parameter ranges
        parameter_ranges = [[parameters['qmri_fit']['parameter_ranges']['D'][0],
                             parameters['qmri_fit']['parameter_ranges']['f'][0],
                             parameters['qmri_fit']['parameter_ranges']['Dp'][0],
                             parameters['qmri_fit']['parameter_ranges']['S0'][0]],
                            [parameters['qmri_fit']['parameter_ranges']['D'][1],
                             parameters['qmri_fit']['parameter_ranges']['f'][1],
                             parameters['qmri_fit']['parameter_ranges']['Dp'][1],
                             parameters['qmri_fit']['parameter_ranges']['S0'][1]]]

        # fit data to IVIM model
        prediction['D'], prediction['f'], prediction['Dp'], prediction['S0'] = fit_least_squares_array_ivim(
            xvals[0, :].squeeze(), signals, S0_output=True, fitS0=True, bounds=parameter_ranges)

        # reconstruct signal from predicted parameters
        signal_pred = ivim_parameters_to_signal(torch.tensor(prediction['D']), torch.tensor(prediction['Dp']),
                                                torch.tensor(prediction['f']), torch.tensor(prediction['S0']),
                                                torch.tensor(np.transpose(xvals)))

        # store predicted signal in prediction dict
        prediction['signal_pred'] = signal_pred.cpu().detach().numpy().transpose()

    elif parameters['qmri_fit']['model'] == 'VFA':
        # set parameter ranges
        parameter_ranges = [[parameters['qmri_fit']['parameter_ranges']['M0'][0],
                             parameters['qmri_fit']['parameter_ranges']['T1'][0]],
                            [parameters['qmri_fit']['parameter_ranges']['M0'][1],
                             parameters['qmri_fit']['parameter_ranges']['T1'][1]]]

        # retrieve TR values from non_predicted_parameters
        TR = torch.stack(non_predicted_parameters).cpu().detach().numpy().squeeze()

        # fit data to VFA model
        prediction['M0'], prediction['T1'] = fit_least_squares_array_vfa(
            xvals[0, :].squeeze(), signals, 4, parameter_ranges, TR)

        # reconstruct signal from predicted parameters
        _, signal_pred = vfa_parameters_to_signal(torch.tensor(prediction['M0']), torch.tensor(prediction['T1']),
                                                  torch.tensor(TR),
                                                  torch.tensor(np.transpose(xvals)))

        # store predicted signal in prediction dict
        prediction['signal_pred'] = np.transpose(signal_pred)

    elif parameters['qmri_fit']['model'] == 'DCE':

        # set parameter ranges
        parameter_ranges = [[parameters['qmri_fit']['parameter_ranges']['ke'][0],
                             parameters['qmri_fit']['parameter_ranges']['dt'][0],
                             parameters['qmri_fit']['parameter_ranges']['ve'][0],
                             parameters['qmri_fit']['parameter_ranges']['vp'][0]],
                            [parameters['qmri_fit']['parameter_ranges']['ke'][1],
                             parameters['qmri_fit']['parameter_ranges']['dt'][1],
                             parameters['qmri_fit']['parameter_ranges']['ve'][1],
                             parameters['qmri_fit']['parameter_ranges']['vp'][1]]]

        # set AIF parameters
        aif = parameters['qmri_fit']['aif']['aif'].copy()
        Hct = np.ones((signals.shape[1])) * parameters['qmri_fit']['aif']['Hct']
        aif['ab'] /= (1 - Hct)

        # fit data to DCE model
        prediction['ke'], prediction['dt'], prediction['ve'], prediction['vp'] = \
            fit_tofts_model(signals, xvals[0, :].squeeze(), aif, bounds=parameter_ranges)

        # reconstruct signal from predicted parameters
        signal_pred = np.zeros(signals.shape)
        for el in range(signal_pred.shape[0]):
            signal_pred[el, :] = Cosine8AIF_ExtKety(xvals[0, :].squeeze(), aif, prediction['ke'][el], prediction['dt'][el],
                                                    prediction['ve'][el], prediction['vp'][el])

        # store predicted signal in prediction dict
        prediction['signal_pred'] = signal_pred

    elif parameters['qmri_fit']['model'] == 'T2star':
        # set parameter ranges
        parameter_ranges = [[parameters['qmri_fit']['parameter_ranges']['T2star'][0],
                             parameters['qmri_fit']['parameter_ranges']['S0'][0]],
                            [parameters['qmri_fit']['parameter_ranges']['T2star'][1],
                             parameters['qmri_fit']['parameter_ranges']['S0'][1]]]

        # fit data to T2* model
        prediction['T2star'], prediction['S0'] = fit_least_squares_array_t2star(xvals[0, :].squeeze(), signals, 4,
                                                                                parameter_ranges)

        # reconstruct signal from predicted parameters
        signal_pred = T2star_parameters_to_signal(torch.tensor(prediction['T2star']), torch.tensor(prediction['S0']),
                                                  torch.tensor(np.transpose(xvals)))

        # store predicted signal in prediction dict
        prediction['signal_pred'] = np.transpose(signal_pred)
    else:
        raise NotImplementedError('other models than IVIM, VFA, T2* and DCE not yet implemented')

    return prediction, signals, signal_pred
