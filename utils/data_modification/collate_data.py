"""
This implementation is partially inspired by examples from torchcde
# (https://github.com/patrick-kidger/torchcde), licensed under Apache 2.0.
"""

import numpy as np
import random
import torch
import torchcde
import logging

from utils.data_modification.irregular_data import prepare_multi_channel_observations, fill_forward, batch_elements


class DataCollater:
    """
    Class for collating data
    """
    def __init__(self, qmri_model):
        """
        Constructor for DataCollater
        Args:
            qmri_model: type of qmri model
        """
        self.qmri_model = qmri_model

    def collate_data(self, batch: list):
        """
        Collate all data in batch of subjects
        Args:
            batch: list of subjects

        Returns:
            new_batch: collated data
            coeffs: coefficients
            non_predicted_parameters: non predicted parameters

        """
        signals, xvals, non_predicted_parameters = [], [], []
        for el in batch:
            signals.append(torch.Tensor(el['signals']))
            xvals.append(torch.Tensor(el['xvals']))

            if self.qmri_model == 'IVIM':
                non_predicted_parameters = None

            elif self.qmri_model == 'VFA':
                non_predicted_parameters.append(torch.Tensor(el['non_predicted_parameters']))

            elif self.qmri_model == 'DCE':
                non_predicted_parameters.append(torch.Tensor(el['non_predicted_parameters']))

        observations = []
        for xval, signal in zip(xvals, signals):
            observation = prepare_multi_channel_observations([[xval], [signal]])
            observations.append(observation)
        filled_x = fill_forward(observations)
        new_batch = batch_elements(filled_x)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(new_batch).float()

        return new_batch, coeffs, non_predicted_parameters

    def random_collate_data(self, batch: list):
        """
        Randomly collate data in batch of subjects, used for training
        Args:
            batch: list of subjects

        Returns:
            new_batch: collated data
            coeffs: coefficients
            non_predicted_parameters: non predicted parameters
        """

        signals, xvals, parameter_coeffs, non_predicted_parameters, metadata = [], [], [], [], []
        for (el, el_metadata) in batch:

            try:
                mask = el.get('valid_mask').numpy()
            except AttributeError:
                logging.info('No valid mask found, using all data')
                mask = np.ones_like(el.get('signals').numpy()[0:1, ...])

            nonzero_indices = np.argwhere(mask[0, ...])
            if nonzero_indices.size == 0:
                logging.error(
                    f'Could not find valid datapoint, max value in array is {np.max(mask)}, moving to next subject')
                continue
            else:
                # Randomly select a valid index from the non-zero indices
                idx = random.choice(nonzero_indices)
                x, y, z = idx[0], idx[1], idx[2]

                signals.append(torch.Tensor(el.get('signals').numpy()[:, x:x+1, y:y+1, z:z+1].squeeze()))
                xvals.append(torch.Tensor(el.get('xvals').numpy()[:, 0, 0, 0].squeeze()))
                metadata.append(el_metadata)

                if self.qmri_model == 'IVIM':
                    non_predicted_parameters = None
                    try:
                        parameter_coeffs.append(torch.stack(
                            [torch.Tensor(el.get('D_coeff_image').numpy()[:, x:x+1, y:y+1, z:z+1].squeeze()),
                             torch.Tensor(el.get('Dp_coeff_image').numpy()[:, x:x+1, y:y+1, z:z+1].squeeze()),
                             torch.Tensor(el.get('f_coeff_image').numpy()[:, x:x+1, y:y+1, z:z+1].squeeze()),
                             torch.Tensor(el.get('S0_coeff_image').numpy()[:, x:x+1, y:y+1, z:z+1].squeeze())]))
                    except AttributeError:
                        parameter_coeffs.append(torch.tensor([0, 0, 0, 0]))

                elif self.qmri_model == 'VFA':
                    non_predicted_parameters.append(torch.stack(
                        [torch.Tensor(el.get('TR_image').numpy()[:, x:x + 1, y:y + 1, z:z + 1].squeeze())]))

                    try:
                        parameter_coeffs.append(torch.stack(
                            [torch.Tensor(el.get('M0_coeff_image').numpy()[:, x:x+1, y:y+1, z:z+1].squeeze()),
                             torch.Tensor(el.get('T1_coeff_image').numpy()[:, x:x+1, y:y+1, z:z+1].squeeze())]))
                    except AttributeError:
                        parameter_coeffs.append(torch.tensor([0, 0]))

                elif self.qmri_model == 'DCE':
                    try:
                        non_predicted_parameters.append(torch.stack(
                            [torch.Tensor(el.get('AIF_image').numpy()[:, x:x + 1, y:y + 1, z:z + 1].squeeze())]))
                    except AttributeError:
                        non_predicted_parameters.append(torch.tensor([float('nan')]))

                    try:
                        parameter_coeffs.append(torch.stack(
                            [torch.Tensor(el.get('ke_coeff_image').numpy()[:, x:x + 1, y:y + 1, z:z + 1].squeeze()),
                            torch.Tensor(el.get('ve_coeff_image').numpy()[:, x:x + 1, y:y + 1, z:z + 1].squeeze()),
                            torch.Tensor(el.get('vp_coeff_image').numpy()[:, x:x + 1, y:y + 1, z:z + 1].squeeze()),
                            torch.Tensor(el.get('dt_coeff_image').numpy()[:, x:x + 1, y:y + 1, z:z + 1].squeeze())]))
                    except AttributeError:
                        parameter_coeffs.append(torch.tensor([0, 0, 0, 0]))

        observations = []

        for xval, signal in zip(xvals, signals):
            observation = prepare_multi_channel_observations([[xval], [signal]])
            observations.append(observation)

        filled_x = fill_forward(observations)
        new_batch = batch_elements(filled_x)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(new_batch).float()
        parameter_coeffs = torch.stack(parameter_coeffs, dim=0)

        if self.qmri_model != 'IVIM':
            non_predicted_parameters = torch.stack(non_predicted_parameters, dim=0)

        return new_batch, coeffs, parameter_coeffs, non_predicted_parameters, metadata
