import random

import numpy as np
import torch
import torchio

from qmri.simulation.ivim_simulation import create_list_of_simulated_ivim_signals, create_list_of_random_bval_arrays
from qmri.simulation.vfa_simulation import create_list_of_simulated_vfa_signals, create_list_of_random_flipangle_arrays
from qmri.simulation.dce_simulation import dce_simulation, create_list_of_timepoint_arrays


def create_structured_simulation_data(simulation_parameters, samples, seed=None):
    """
    loads structured (4D volumes) simulation data
    Args:
        simulation_parameters: dictionary containing simulation parameters
        samples: number of samples to be generated
        seed: seed for random number generator

    Returns:
        simulation_set, a list of Subjects

    """
    if seed is not None:
        random.seed(seed)

    if simulation_parameters['structured']['sort_on_xvals'] and simulation_parameters['structured']['sort_on_SNR']:
        structured_SNR = random.choice(np.linspace(simulation_parameters['SNR'][0],
                                                   simulation_parameters['SNR'][1], dtype='int'))
        structured_SNRs = [structured_SNR, structured_SNR + 1]

        if simulation_parameters['qmri'] == 'IVIM':
            if 'fixed' in simulation_parameters.keys() and simulation_parameters['fixed']:
                fixed_list = [simulation_parameters['bvalues']]
                nb_lists = (simulation_parameters['structured']['samples_per_patient'] *
                            simulation_parameters['structured']['nb_patients'])
                structured_xvals = [el for el in fixed_list for _ in range(nb_lists)]
            else:
                structured_xvals = create_list_of_random_bval_arrays(simulation_parameters['structured']['nb_patients'],
                                                                     simulation_parameters['bvalues']['size'],
                                                                     simulation_parameters['bvalues']['sample_set'],
                                                                     seed)
                structured_xvals_list = []
                for i in range(len(structured_xvals)):
                    structured_xvals_list.append([structured_xvals[i]]*simulation_parameters['structured']['samples_per_patient'])
                structured_xvals = [item for sublist in structured_xvals_list for item in sublist]

            unstructured_sim = create_list_of_simulated_ivim_signals(SNR_range=structured_SNRs,
                                                                     samples=samples,
                                                                     parameter_ranges=simulation_parameters[
                                                                             'parameter_ranges'],
                                                                     list_of_bvalues=structured_xvals,
                                                                     seed_nr=seed)
        elif simulation_parameters['qmri'] == 'VFA':
            if 'fixed' in simulation_parameters.keys() and simulation_parameters['fixed']:
                fixed_list = [simulation_parameters['flip_angles']]
                nb_lists = (simulation_parameters['structured']['samples_per_patient'] *
                            simulation_parameters['structured']['nb_patients'])
                structured_xvals = [el for el in fixed_list for _ in range(nb_lists)]
            else:
                structured_xvals = create_list_of_random_flipangle_arrays(
                    simulation_parameters['structured']['nb_patients'], simulation_parameters['flip_angles']['size'],
                    simulation_parameters['flip_angles']['sample_set'], seed)
                structured_xvals_list = []
                for i in range(len(structured_xvals)):
                    structured_xvals_list.append([structured_xvals[i]]*simulation_parameters['structured']['samples_per_patient'])
                structured_xvals = [item for sublist in structured_xvals_list for item in sublist]
            unstructured_sim = create_list_of_simulated_vfa_signals(SNR_range=structured_SNRs,
                                                                    samples=samples,
                                                                    parameter_ranges=simulation_parameters[
                                                                            'parameter_ranges'],
                                                                    list_of_flip_angles=structured_xvals,
                                                                    seed_nr=seed)
        elif simulation_parameters['qmri'] == 'DCE':
            if 'fixed' in simulation_parameters.keys() and simulation_parameters['fixed']:
                fixed_list = [simulation_parameters['timepoints']]
                nb_lists = (simulation_parameters['structured']['samples_per_patient'] *
                            simulation_parameters['structured']['nb_patients'])
                structured_xvals = [np.array(el) for el in fixed_list for _ in range(nb_lists)]
            else:
                structured_xvals = create_list_of_timepoint_arrays(
                    simulation_parameters['structured']['nb_patients'],
                    simulation_parameters['timepoints']['temporal_resolution_range'],
                    simulation_parameters['acquisition']['nr_of_measurements_range'],
                     seed)
                structured_xvals_list = []
                for i in range(len(structured_xvals)):
                    structured_xvals_list.append([structured_xvals[i]]*simulation_parameters['structured']['samples_per_patient'])
                structured_xvals = [item for sublist in structured_xvals_list for item in sublist]
            unstructured_sim = dce_simulation(simulation_parameters, structured_xvals, seed)
            structured_xvals = [el['xvals'] for el in unstructured_sim]
        else:
            raise NotImplementedError

        study_dict = {}
        for i in range(simulation_parameters['structured']['nb_patients']):
            subject_dict = {}
            start_sample = i * simulation_parameters['structured']['samples_per_patient']
            end_sample = (i+1) * simulation_parameters['structured']['samples_per_patient']
            unstructured_sim_subject = unstructured_sim[start_sample:end_sample]
            nb_xvals = len(structured_xvals[start_sample])
            if simulation_parameters['qmri'] == 'IVIM':
                subject_dict['D_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['D'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['Dp_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['Dp'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['f_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['f'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['S0_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['S0'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['D_coeff_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['D_coeff'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['Dp_coeff_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['Dp_coeff'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['f_coeff_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['f_coeff'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['S0_coeff_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['S0_coeff'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['signals'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.transpose(np.stack([el['signals'] for el in unstructured_sim_subject])),
                               (nb_xvals, *simulation_parameters['structured']['shape']))))
            elif simulation_parameters['qmri'] == 'VFA':
                subject_dict['M0_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['M0'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['T1_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['T1'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['TR_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['TR'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['M0_coeff_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['M0_coeff'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['T1_coeff_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['T1_coeff'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['TR_coeff_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['TR_coeff'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['signals'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.transpose(np.stack([el['signals'] for el in unstructured_sim_subject])),
                               (nb_xvals, *simulation_parameters['structured']['shape']))))
            elif simulation_parameters['qmri'] == 'DCE':
                subject_dict['ve_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['ve'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['vp_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['vp'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['ke_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['ke'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['dt_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['dt'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['ve_coeff_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['ve_coeff'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['vp_coeff_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['vp_coeff'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['ke_coeff_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['ke_coeff'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['dt_coeff_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['dt_coeff'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['Hct_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.stack([el['Hct'] for el in unstructured_sim_subject]),
                               (1, *simulation_parameters['structured']['shape']))))
                subject_dict['AIF_image'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.transpose(np.stack([el['aif'] for el in unstructured_sim_subject])),
                               (9, *simulation_parameters['structured']['shape']))))
                subject_dict['signals'] = torchio.Image(tensor=torch.Tensor(
                    np.reshape(np.transpose(np.stack([el['signals'] for el in unstructured_sim_subject])),
                               (len(unstructured_sim_subject[0]['xvals']),
                                *simulation_parameters['structured']['shape']))))
            else:
                raise NotImplementedError
            subject_dict['xvals'] = torchio.Image(tensor=torch.Tensor(
                np.reshape(unstructured_sim_subject[0]['xvals'], (len(unstructured_sim_subject[0]['xvals']), 1, 1, 1))))
            study_dict[i] = (torchio.Subject(subject_dict))
    else:
        raise NotImplementedError

    return study_dict


def create_unstructured_simulation_data(simulation_parameters, samples, seed):
    """
    loads structured (4D volumes) simulation data
    Args:
        seed: seed for random number generator
        simulation_parameters: dictionary containing simulation parameters
        samples: number of samples to be generated

    Returns: simulation_set, a list of Subjects

    """
    if simulation_parameters['qmri'] == 'IVIM':
        # generate b values at which data needs to be simulated
        if isinstance(simulation_parameters['bvalues'], dict):
            bvalue_lists = create_list_of_random_bval_arrays(samples, simulation_parameters['bvalues']['size'],
                                                             simulation_parameters['bvalues']['sample_set'],
                                                             seed)
        else:
            bvalue_lists = [simulation_parameters['bvalues']] * samples

        simulation_set = create_list_of_simulated_ivim_signals(SNR_range=simulation_parameters['SNR'],
                                                               samples=samples,
                                                               parameter_ranges=simulation_parameters['parameter_ranges'],
                                                               list_of_bvalues=bvalue_lists,
                                                               seed_nr=seed)
    elif simulation_parameters['qmri'] == 'VFA':
        # generate flip angles at which data needs to be simulated
        if isinstance(simulation_parameters['flip_angles'], dict):
            flip_angle_lists = create_list_of_random_flipangle_arrays(samples, simulation_parameters['flip_angles']['size'],
                                                                      simulation_parameters['flip_angles']['sample_set'],
                                                                      seed)
        else:
            flip_angle_lists = [simulation_parameters['flip_angles']] * samples

        simulation_set = create_list_of_simulated_vfa_signals(SNR_range=simulation_parameters['SNR'],
                                                              samples=samples,
                                                              parameter_ranges=simulation_parameters[
                                                                  'parameter_ranges'],
                                                              list_of_flip_angles=flip_angle_lists,
                                                              seed_nr=seed)

    if simulation_parameters['seed_nr'] is None:
        random.shuffle(simulation_set)
    else:
        random.Random(seed).shuffle(simulation_set)

    return simulation_set
