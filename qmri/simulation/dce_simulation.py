import random
import numpy as np
import qmri.forward_model.dce as dce
from tqdm import tqdm


def dce_simulation(simulation_parameters, list_of_xval_arrays, seed):
    """
    simulate DCE data
    Args:
        simulation_parameters: parameters for simulation
        list_of_xval_arrays: list of timepoint arrays
        seed: random seed

    Returns:
        data_list: list of simulated data

    """
    SNR = random.choice(np.linspace(simulation_parameters['SNR'][0],
                                    simulation_parameters['SNR'][1], dtype='int'))
    Hct = simulation_parameters['aif']['Hct']
    vp_range = simulation_parameters['parameter_ranges']['vp']
    ve_range = simulation_parameters['parameter_ranges']['ve']
    kep_range = simulation_parameters['parameter_ranges']['ke']
    dt_range = simulation_parameters['parameter_ranges']['dt']
    samples = simulation_parameters['samples']

    if 'noise_method' in simulation_parameters.keys():
        noise_method = simulation_parameters['noise_method']
    else:
        noise_method = 'parameter_independent'

    if 'sampling_method' in simulation_parameters.keys():
        sampling_method = simulation_parameters['sampling_method']
    else:
        sampling_method = 'uniform'

    # random number generator
    if seed is None:
        rg = np.random.RandomState()
    else:
        rg = np.random.RandomState(seed)

    # generate parameter coefficients
    if sampling_method == 'uniform':
        ke_coeff = rg.uniform(0, 1, (samples, 1))
        ve_coeff = rg.uniform(0, 1, (samples, 1))
        vp_coeff = rg.uniform(0, 1, (samples, 1))
        dt_coeff = rg.uniform(0, 1, (samples, 1))
    elif sampling_method == 'gaussian':
        ke_coeff = rg.normal(0.5, 0.15, (samples, 1))
        ve_coeff = rg.normal(0.5, 0.15, (samples, 1))
        vp_coeff = rg.normal(0.5, 0.15, (samples, 1))
        dt_coeff = rg.normal(0.5, 0.15, (samples, 1))
        ke_coeff[ke_coeff < 0] = 0
        ke_coeff[ke_coeff > 1] = 1
        ve_coeff[ve_coeff < 0] = 0
        ve_coeff[ve_coeff > 1] = 1
        vp_coeff[vp_coeff < 0] = 0
        vp_coeff[vp_coeff > 1] = 1

    ke = kep_range[0] + (ke_coeff * (kep_range[1] - kep_range[0]))
    ve = ve_range[0] + (ve_coeff * (ve_range[1] - ve_range[0]))
    vp = vp_range[0] + (vp_coeff * (vp_range[1] - vp_range[0]))
    dt = dt_range[0] + (dt_coeff * (dt_range[1] - dt_range[0]))

    # vary the Hct value from 0.3 to 0.6
    aif = simulation_parameters['aif']['aif'].copy()
    aif['ab'] /= (1 - Hct)
    aif_arr = np.array([aif['t0'], aif['tr'], aif['ab'], aif['mb'], aif['ae'], aif['me'], aif['ar'], aif['mm'],
                        aif['mr']])

    if noise_method == 'parameter_independent':
        # calculate average values for the parameters for SNR noise calculation
        average_ke = np.mean(ke)
        average_ve = np.mean(ve)
        average_vp = np.mean(vp)
        average_dt = np.mean(dt)

    data_list = []
    for i in tqdm(range(samples)):
        simulated_data = dce.Cosine8AIF_ExtKety(
           list_of_xval_arrays[i],
           aif,
           ke[i][0],
           dt[i][0],
           ve[i][0], vp[i][0])

        if noise_method == 'parameter_independent':
            average_simulated_data = dce.Cosine8AIF_ExtKety(
              list_of_xval_arrays[i], aif, average_ke, average_dt, average_ve, average_vp)
            signal_level = np.max(average_simulated_data)
        elif noise_method == 'parameter_dependent':
            signal_level = np.max(simulated_data)

        noise_real = rg.normal(0, signal_level / SNR, (1, len(list_of_xval_arrays[i])))
        noisy_data = np.add(simulated_data, noise_real).squeeze()

        el_dict = {'signals': noisy_data.squeeze(), 'xvals': list_of_xval_arrays[i], 've': ve[i][0],
                   've_coeff': ve_coeff[i][0], 'vp': vp[i][0], 'vp_coeff': vp_coeff[i][0], 'ke': ke[i][0],
                   'ke_coeff': ke_coeff[i][0], 'dt': dt[i][0], 'dt_coeff': dt_coeff[i][0], 'Hct': Hct, 'aif': aif_arr}
        data_list.append(el_dict)
    return data_list


def create_list_of_timepoint_arrays(samples, temporal_resolution_range, nr_of_measurements_range, seed_nr=None):
    """
    creates list of timepoint arrays of bound length based on a set of xvalues to sample from
    Args:
        samples: int amount of xvalue arrays
        temporal_resolution_range:
        nr_of_measurements_range:
        seed_nr: int or None, whether the result should be reproducible or not

    Returns:
        list_of_xval_arrays: list of timepoint arrays

    """

    # choose a random number generator
    if seed_nr is None:
        rg = np.random.RandomState()
    else:
        rg = np.random.RandomState(seed_nr)

    # generate temporal resolution
    temporal_resolution = rg.uniform(temporal_resolution_range[0], temporal_resolution_range[1], samples).astype(int)
    nr_of_measurements = rg.uniform(nr_of_measurements_range[0], nr_of_measurements_range[1], samples).astype(int)

    # create simulation parameters
    list_of_xval_arrays = [range(1, nr_of_measurements[a]+1) * temporal_resolution[a] / 60 for a in range(samples)]

    return list_of_xval_arrays
