import torch
import numpy as np

from qmri.forward_model.ivim import ivim_parameters_to_signal
from qmri.simulation.SNR import create_array_of_SNR_values


def create_list_of_simulated_ivim_signals(SNR_range, list_of_bvalues, samples, parameter_ranges,
                                          sampling_method='uniform', seed_nr=None):
    """
    create list of simulated ivim signals
    Args:
        SNR_range: list with 2 entries, indicating lower and upper bound of SNR
        list_of_bvalues: list of b-values at which ivim signal needs to be simulated
        samples: amount of ivim decay curves
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

    # Generate D, Dp (=D*), f and S0 constraints
    Dmin, Dmax = parameter_ranges['D'][0], parameter_ranges['D'][1]
    fmin, fmax = parameter_ranges['f'][0], parameter_ranges['f'][1]
    Dpmin, Dpmax = parameter_ranges['Dp'][0], parameter_ranges['Dp'][1]
    S0min, S0max = parameter_ranges['S0'][0], parameter_ranges['S0'][1]

    # generate parameter coefficients
    if sampling_method == 'uniform':
        D_coeff = rg.uniform(0, 1, (samples, 1))
        Dp_coeff = rg.uniform(0, 1, (samples, 1))
        f_coeff = rg.uniform(0, 1, (samples, 1))
        S0_coeff = np.expand_dims(np.array([0.5] * samples), axis=1)
    elif sampling_method == 'gaussian':
        D_coeff = rg.normal(0.5, 0.15, (samples, 1))
        Dp_coeff = rg.normal(0.5, 0.15, (samples, 1))
        f_coeff = rg.normal(0.5, 0.15, (samples, 1))
        S0_coeff = np.expand_dims(np.array([0.5] * samples), axis=1)
        D_coeff[D_coeff < 0] = 0
        D_coeff[D_coeff > 1] = 1
        Dp_coeff[Dp_coeff < 0] = 0
        Dp_coeff[Dp_coeff > 1] = 1
        f_coeff[f_coeff < 0] = 0
        f_coeff[f_coeff > 1] = 1

    # convert parameter coefficients to parameters
    D = Dmin + (D_coeff * (Dmax - Dmin))
    Dp = Dpmin + (Dp_coeff * (Dpmax - Dpmin))
    f = fmin + (f_coeff * (fmax - fmin))
    S0 = S0min + (S0_coeff * (S0max - S0min))

    # create SNR array
    list_of_SNR_arrays = create_array_of_SNR_values(SNR_range, samples, rg)

    # loop over array to fill with simulated IVIM data
    data = []
    for i in range(samples):
        data_point = {}
        simulated_data = simulate_ivim_signal(D[i][0], Dp[i][0], f[i][0], S0[i][0], list_of_bvalues[i],
                                              list_of_SNR_arrays[i], rg)

        # add parameters to data
        data_point["D"] = D[i][0]
        data_point["f"] = f[i][0]
        data_point["Dp"] = Dp[i][0]
        data_point['S0'] = S0[i][0]
        data_point["D_coeff"] = D_coeff[i][0]
        data_point["f_coeff"] = f_coeff[i][0]
        data_point["Dp_coeff"] = Dp_coeff[i][0]
        data_point["S0_coeff"] = S0_coeff[i][0]

        # add signal to data
        data_point["xvals"] = [bval for bval in list_of_bvalues[i]]
        data_point["signals"] = simulated_data.squeeze()
        data.append(data_point)
    return data


def simulate_ivim_signal(D, Dp, f, S0, bvalues, SNR_array, rg):
    """
    simulate ivim signal

    Args:
        D: diffusion coefficient
        Dp: pseudo diffusion coefficient
        f: perfusion fraction
        S0: signal without diffusion weighting
        bvalues: b-values (measure of diffusion weighting)
        SNR_array: noise to be added to the simulated data
        rg: random number generator

    Returns:
        simulated_data: simulated ivim signal

    """
    bvalues.sort()
    b0_bool = np.array(bvalues) == 0
    simulated_data = ivim_parameters_to_signal(torch.tensor(D), torch.tensor(Dp), torch.tensor(f), torch.tensor(S0),
                                               torch.tensor(np.asarray(bvalues)))
    simulated_data = simulated_data.cpu().detach().numpy()

    # create 2 signal arrays filled with gaussian noise
    noise_real = rg.normal(0, 1 / SNR_array, (1, len(bvalues)))
    noise_imag = rg.normal(0, 1 / SNR_array, (1, len(bvalues)))

    # add Rician noise to the simulated data
    simulated_data = np.sqrt(np.power(simulated_data + noise_real, 2) + np.power(noise_imag, 2)).squeeze()

    # renormalize simulated data to noisy S0
    S0_noisy = np.mean(simulated_data[b0_bool])
    simulated_data /= S0_noisy
    return simulated_data


def create_list_of_random_bval_arrays(samples, bounds_subset_size, xval_set, seed_nr=None):
    """
    creates list of random xvalue arrays of bounded length based on a set of xvalues to sample from
    Args:
        samples: int amount of xvalue arrays
        bounds_subset_size: list containing min and max length of xvalue array
        xval_set: list, set to sample bvalues from
        seed_nr: int or None, whether the result should be reproducible or not

    Returns:
        list_of_xval_arrays: list of bvalue arrays

    """
    list_of_xval_arrays = []

    # choose a random number generator
    if seed_nr is None:
        rg = np.random.RandomState()
    else:
        rg = np.random.RandomState(seed_nr)

    for i in range(samples):
        size = rg.randint(bounds_subset_size[0], bounds_subset_size[1])
        size_below_bval100  = int(np.ceil(size / 2))
        size_below_bval50 = int(np.ceil(size_below_bval100 / 2))
        size_between_bval_50_100 = size_below_bval100 - size_below_bval50
        size_above_bval100 = size - size_below_bval100
        xval_set_below_bval50 = [xval for xval in xval_set if xval < 50]
        xval_set_between_bval50_100 = [xval for xval in xval_set if xval >= 50 and xval < 100]
        xval_set_above_bval100 = [xval for xval in xval_set if xval >= 100]
        xval_below_bval50 = rg.choice(xval_set_below_bval50, size_below_bval50, replace=False)
        xval_between_bval50_100 = rg.choice(xval_set_between_bval50_100, size_between_bval_50_100, replace=False)
        xval_above_bval100 = rg.choice(xval_set_above_bval100, size_above_bval100, replace=False)
        xval = np.concatenate([np.array([0]), xval_below_bval50, xval_between_bval50_100, xval_above_bval100])
        args = np.argsort(xval)
        xval = xval[args]
        xval_unique = np.asarray(list(set(xval)))
        xval_unique.sort()
        list_of_xval_arrays.append(xval_unique)
    return list_of_xval_arrays


