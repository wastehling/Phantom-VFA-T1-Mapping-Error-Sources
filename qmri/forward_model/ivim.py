import torch


def ivim_coefficients_to_parameters(coefficients, parameter_ranges, mapping_method='sigmoid'):
    """
    converts ivim coefficients to parameters (D, Dp, f, S0)
    Args:
        coefficients: input coefficients
        parameter_ranges: ranges of output parameters to which input coefficients should be mapped
        mapping_method: method of mapping

    Returns:
        D, Dp, f, S0: ivim parameters

    """

    # retrieve parameter constraints
    Dmin, Dmax = parameter_ranges['D'][0], parameter_ranges['D'][1]
    Dpmin, Dpmax = parameter_ranges['Dp'][0], parameter_ranges['Dp'][1]
    fmin, fmax = parameter_ranges['f'][0], parameter_ranges['f'][1]
    S0min, S0max = parameter_ranges['S0'][0], parameter_ranges['S0'][1]

    # coefficient to parameter
    if mapping_method == 'sigmoid':
        D = Dmin + torch.sigmoid(coefficients[:, 0].unsqueeze(1)) * (Dmax - Dmin)
        Dp = Dpmin + torch.sigmoid(coefficients[:, 1].unsqueeze(1)) * (Dpmax - Dpmin)
        f = fmin + torch.sigmoid(coefficients[:, 2].unsqueeze(1)) * (fmax - fmin)
        S0 = S0min + torch.sigmoid(coefficients[:, 3].unsqueeze(1)) * (S0max - S0min)
    elif mapping_method == 'linear':
        D = Dmin + (coefficients[:, 0].unsqueeze(1) * (Dmax - Dmin))
        Dp = Dpmin + (coefficients[:, 1].unsqueeze(1) * (Dpmax - Dpmin))
        f = fmin + (coefficients[:, 2].unsqueeze(1) * (fmax - fmin))
        S0 = S0min + (coefficients[:, 3].unsqueeze(1) * (S0max - S0min))
    else:
        raise NotImplementedError(f'only linear and sigmoidal simulation method available, not {mapping_method}')

    return D, Dp, f, S0


def ivim_parameters_to_signal(D, Dp, f, S0, bvalues):
    """
    converts ivim parameters to predicted signal at specified bvalues
    Args:
        D: diffusion coefficient
        Dp: pseudo diffusion coefficient
        f: perfusion fraction
        S0: signal at b=0
        bvalues: b-values (measures of diffusion weighting)

    Returns:
        signal: signal at specified b-values

    """
    # Calculate signal based on estimated parameters
    relative_signal = f * torch.exp(-bvalues * Dp) + (1 - f) * torch.exp(-bvalues * D)
    signal = S0 * relative_signal
    return signal
