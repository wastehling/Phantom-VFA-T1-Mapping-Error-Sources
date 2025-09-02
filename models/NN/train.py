import os
import torch
import logging
import numpy as np
import copy

from tqdm import tqdm
from models.NN.validation import validation_NN
from qmri.forward_model.ivim import ivim_coefficients_to_parameters, ivim_parameters_to_signal
from qmri.forward_model.vfa import (vfa_coefficients_to_parameters, vfa_parameters_to_signal,
                                    vfa_parameters_to_signal_transformed)
from qmri.forward_model.dce import dce_coefficients_to_parameters, dce_parameters_to_signal
from utils.save_data import save_loss_history


def train_NN(model, parameters, train_loader, val_loader, metric_list, target_folder, device):
    """
    train function of model based on data from train_loader, val_loader and sim_loader
    Args:
        model: model object to train
        parameters: parameters dict
        train_loader: dataloader object containing in vivo train data
        val_loader: dataloader object containing in vivo val data
        metric_list: list of metrics to monitor perform
        target_folder: where to store model
        device: either cuda or cpu

    Returns:
        model: trained model object
        loss_dict: dict of loss during training

    """

    # make directory for model checkpoints
    os.makedirs(target_folder + '/model_checkpoints/')

    # initialize losses
    signal_losses, parameter_losses, total_losses, val_signal_losses, val_parameter_losses, val_total_losses = \
        [], [], [], [], [], []

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['training']['learning_rate'])
    if 'scheduler' in parameters['training'].keys():
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=parameters['training']['scheduler']['milestones'],
                                                         gamma=parameters['training']['scheduler']['gamma'])
    else:
        scheduler = None

    if parameters['training']['loss'] == 'MSE':
        criterion = torch.nn.MSELoss(reduction='mean').to(device)
    else:
        raise NotImplementedError('other losses not implemented')

    steps_per_epoch = parameters['training']['steps_per_epoch']

    # Training loop
    epochs = parameters['training']['epochs']
    for epoch in range(1, epochs+1):
        if epoch % parameters['training']['model_checkpoint_interval'] == 0:
            # save model
            logging.info(f'storing model checkpoint at epoch {epoch}')
            torch.save(model, target_folder + f'/model_checkpoints/model_{epoch}.pt')

        steps = 0
        cum_signal_loss = 0
        cum_parameter_loss = 0
        cum_total_loss = 0
        logging.info(f'epoch {epoch} of {epochs}')

        # set model in training mode
        model.train()

        # loop over iterations per epoch
        for data, cubic_coeffs, parameter_coeffs, non_predicted_parameters in tqdm(train_loader):

            # break out of epoch
            if steps > steps_per_epoch:
                break

            # decompose x and signals component
            x_vals = data[:, :, 0].float().to(device)
            signals = data[:, :, 1].float().to(device)

            # coeffs to device
            parameter_coeffs = parameter_coeffs.float().to(device)
            if non_predicted_parameters is not None:
                non_predicted_parameters = non_predicted_parameters.float().to(device)

            # zero grad optimizer
            optimizer.zero_grad()

            # forward pass
            model_coefficients = model(signals)

            # parameter loss
            if parameters['qmri_fit']['model'] == 'VFA':
                M0, T1overTR = vfa_coefficients_to_parameters(model_coefficients,
                                                              parameters['qmri_fit']['parameter_ranges'],
                                                              parameters['qmri_fit']['mapping_method'])
                TRoverT1 = 1 / T1overTR
                TR = torch.reshape(non_predicted_parameters[:, 0], TRoverT1.size())
                T1 = TR / TRoverT1
                T1min, T1max = parameters['qmri_fit']['parameter_ranges']['T1'][0], parameters['qmri_fit']['parameter_ranges']['T1'][1]
                M0min, M0max = parameters['qmri_fit']['parameter_ranges']['M0'][0], parameters['qmri_fit']['parameter_ranges']['M0'][1]
                T1_linear_coeff = (T1 - T1min) / (T1max - T1min)
                M0_linear_coeff = (M0 - M0min) / (M0max - M0min)
                model_coefficients_linear = torch.cat((M0_linear_coeff, T1_linear_coeff), dim=1)
                parameter_criterion = torch.nn.MSELoss(reduction='none').to(device)
                parameter_loss = parameter_criterion(model_coefficients_linear, parameter_coeffs)
                parameter_coeffs_boolean = parameter_coeffs.bool().int()
                parameter_loss = parameter_coeffs_boolean * parameter_loss
                parameter_loss = torch.mean(parameter_loss)
            else:
                sigmoid_model_coefficients = torch.sigmoid(model_coefficients)
                parameter_criterion = torch.nn.MSELoss(reduction='none').to(device)
                parameter_loss = parameter_criterion(sigmoid_model_coefficients, parameter_coeffs)
                parameter_coeffs_boolean = parameter_coeffs.bool().int()
                parameter_loss = parameter_coeffs_boolean * parameter_loss
                parameter_loss = torch.mean(parameter_loss)

            # signal prediction
            # coefficients to qmri parameters
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
                raise NotImplementedError('other models than IVIM, VFA or DCE not yet implemented')

            # calculate loss
            observations_mask = torch.ones(signals.size(), device=device)
            signals = torch.multiply(signals, observations_mask)
            signal_pred = torch.multiply(signal_pred, observations_mask)

            if parameters['training']['intensity_weighted']:
                signal_diff_per_element = torch.pow(torch.subtract(signal_pred, signals), 2)
                intensity_per_element = torch.mean(signals, dim=1)
                intensity_per_element = intensity_per_element.unsqueeze(-1).repeat(1, signal_diff_per_element.size(-1))
                signal_loss_per_element = torch.sum(signal_diff_per_element, dim=1) / torch.sum(observations_mask, dim=1)
                signal_loss = torch.mean(signal_loss_per_element)
                signal_diff_per_element.register_hook(lambda grad: grad * torch.nan_to_num(1 / intensity_per_element))
            else:
                signal_diff_per_element = torch.pow(torch.subtract(signal_pred, signals), 2)
                signal_loss_per_element = torch.sum(signal_diff_per_element, dim=1) / torch.sum(observations_mask, dim=1)
                signal_loss = torch.mean(signal_loss_per_element)

            # update the model
            total_loss = (1 - parameters['training']['parameter_loss_fraction']) * signal_loss + \
                            parameters['training']['parameter_loss_fraction'] * parameter_loss
            total_loss.backward()
            optimizer.step()

            if scheduler is not None:
                 scheduler.step()

            # compute average loss over epoch
            cum_signal_loss = cum_signal_loss + signal_loss.cpu().detach().numpy()
            cum_parameter_loss = cum_parameter_loss + parameter_loss.cpu().detach().numpy()
            cum_total_loss = cum_total_loss + total_loss.cpu().detach().numpy()
            steps += 1

        # compute signal loss, parameter loss and total loss
        signal_loss = cum_signal_loss/steps
        parameter_loss = cum_parameter_loss/steps
        total_loss = cum_total_loss/steps
        signal_losses.append(signal_loss)
        parameter_losses.append(parameter_loss)
        total_losses.append(total_loss)
        logging.info("train signal loss: " + str(signal_loss))
        logging.info("train parameter loss: " + str(parameter_loss))
        logging.info("train total loss: " + str(total_loss))

        # perform validation
        if val_loader is not None:
            val_signal_loss, val_parameter_loss, val_total_loss, metric_list = validation_NN(model, val_loader,
                                                                                             parameters, metric_list,
                                                                                             device, criterion)
            logging.info("validation signal loss: " + str(val_signal_loss))
            logging.info("validation parameter loss: " + str(val_parameter_loss))
            logging.info("validation total loss: " + str(val_total_loss))
            val_signal_losses.append(val_signal_loss)
            val_parameter_losses.append(val_parameter_loss)
            val_total_losses.append(val_total_loss)

        metric_dict = {'train_signal_losses': signal_losses,
                       'train_parameter_losses': parameter_losses,
                       'train_total_losses': total_losses,
                       'val_signal_losses': val_signal_losses,
                       'val_parameter_losses': val_parameter_losses,
                       'val_total_losses': val_total_losses}

        save_loss_history(target_folder, metric_dict)

    logging.info('training finished')

    for metric in metric_list:
        metric_dict[metric.name] = metric.average_metrics

    return model, metric_dict
