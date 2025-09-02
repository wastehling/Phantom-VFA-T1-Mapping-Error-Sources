import torch
import numpy as np

from qmri.forward_model.vfa import vfa_parameters_to_signal
from qmri.simulation.SNR import create_array_of_SNR_values


def create_list_of_simulated_vfa_signals(SNR_range, list_of_flip_angles, samples, parameter_ranges,
                                         sampling_method='uniform',
                                         seed_nr=None):
    """
    create list of simulated vfa signals
    Args:
        SNR_range: list with 2 entries, indicating lower and upper bound of SNR
        list_of_flip_angles: list of flip angles at which vfa signal needs to be simulated
        samples: amount of VFA curves
        parameter_ranges: ranges between which parameters vary
        sampling_method: how to sample coefficients
        seed_nr: int or None, whether output of simulation should be reproducible

    Returns:
        data: a list of which each entry is a dict with simulated data and the coefficients used for simulation

    """
    # choose a random number generator
    if seed_nr is None:
        rg = np.random.RandomState()
    else:
        rg = np.random.RandomState(seed_nr)

    # Generate M0, TR, T1 constraints
    M0min, M0max = parameter_ranges['M0'][0], parameter_ranges['M0'][1]
    T1min, T1max = parameter_ranges['T1'][0], parameter_ranges['T1'][1]
    TRmin, TRmax = parameter_ranges['TR'][0], parameter_ranges['TR'][1]

    # generate parameter coefficients
    if sampling_method == 'uniform':
        M0_coeff = rg.uniform(0, 1, (samples, 1))
        T1_coeff = rg.uniform(0, 1, (samples, 1))
        TR_coeff = rg.uniform(0, 1, (samples, 1))
    elif sampling_method == 'gaussian':
        M0_coeff = rg.normal(0.5, 0.15, (samples, 1))
        T1_coeff = rg.normal(0.5, 0.15, (samples, 1))
        TR_coeff = rg.normal(0.5, 0.15, (samples, 1))
        M0_coeff[M0_coeff < 0] = 0
        M0_coeff[M0_coeff > 1] = 1
        T1_coeff[T1_coeff < 0] = 0
        T1_coeff[T1_coeff > 1] = 1
        TR_coeff[TR_coeff < 0] = 0
        TR_coeff[TR_coeff > 1] = 1

    # convert parameter coefficients to parameters
    M0 = M0min + (M0_coeff * (M0max - M0min))
    T1 = T1min + (T1_coeff * (T1max - T1min))
    TR = TRmin + (TR_coeff * (TRmax - TRmin))

    # create SNR array
    list_of_SNR_arrays = create_array_of_SNR_values(SNR_range, samples, rg)

    # loop over array to fill with simulated IVIM data
    data = []
    for i in range(samples):
        data_point = {}
        simulated_data = simulate_VFA_signal(M0[i][0], T1[i][0], TR[i][0], list_of_flip_angles[i],
                                             list_of_SNR_arrays[i], 'M0NR', rg)

        # add parameters to data
        data_point["M0"] = M0[i][0]
        data_point["T1"] = T1[i][0]
        data_point["TR"] = TR[i][0]
        data_point["M0_coeff"] = M0_coeff[i][0]
        data_point["T1_coeff"] = T1_coeff[i][0]
        data_point["TR_coeff"] = TR_coeff[i][0]

        # add signal to data
        data_point["xvals"] = [FA for FA in list_of_flip_angles[i]]
        data_point["signals"] = simulated_data.squeeze()
        data.append(data_point)
    return data


def simulate_VFA_signal(M0, T1, TR, FA, SNR_array, noise_method, rg):
    """
    simulate VFA signal (Mz), following:
    Mz(FA) = M0*sin(FA)*(1-e^(-TR/T1))/(1-cos(FA)e^(-TR/T1))
    with addition of Rician noise

    Args:
        M0: magnetization at thermal equilibirum
        T1: array of T1 characteristic
        TR: array of repetition time
        FA: flip angle
        SNR_array: noise to be added to the simulated data
        noise_method: method of adding noise (currently only M0NR is implemented)
        rg: random number generator

    Returns:
        simulated_data: simulated VFA signal

    """
    _, simulated_data = vfa_parameters_to_signal(torch.tensor(M0), torch.tensor(T1), torch.tensor(TR),
                                                 torch.tensor(np.asarray(FA)))
    simulated_data = simulated_data.cpu().detach().numpy()

    if noise_method == 'M0NR':
        noise_real = rg.normal(0, M0 / SNR_array, (1, len(FA)))
        noise_imag = rg.normal(0, M0 / SNR_array, (1, len(FA)))
    else:
        raise NotImplementedError(f'only M0NR noise method available, not {noise_method}')

    # add Rician noise to the simulated data
    simulated_data_noisy = np.sqrt(np.power(simulated_data + noise_real, 2) + np.power(noise_imag, 2)).squeeze()

    return simulated_data_noisy


def create_list_of_random_flipangle_arrays(samples, bounds_subset_size, xval_set, seed_nr=None):
    """
    creates list of random flip angle arrays of bound length based on a set of flip angles to sample from
    Args:
        samples: int amount of xvalue arrays
        bounds_subset_size: list containing min and max length of xvalue array
        xval_set: list, set to sample bvalues from
        seed_nr: int or None, whether the result should be reproducible or not

    Returns:
        list of xvalue arrays

    """
    list_of_xval_arrays = []

    # choose a random number generator
    if seed_nr is None:
        rg = np.random.RandomState()
    else:
        rg = np.random.RandomState(seed_nr)

    for i in range(samples):
        size = rg.randint(bounds_subset_size[0], bounds_subset_size[1])
        smallest_flip_angle = rg.choice([1, 2])
        large_flip_angle = rg.randint(20, 30)
        size_body = size - 2
        xval_body = rg.choice(xval_set, size_body, replace=False)
        xval = np.concatenate([np.array([smallest_flip_angle]), xval_body, np.array([large_flip_angle])])
        args = np.argsort(xval)
        xval = xval[args]
        xval_unique = np.asarray(list(set(xval)))
        xval_unique.sort()
        list_of_xval_arrays.append(xval_unique)
    return list_of_xval_arrays
