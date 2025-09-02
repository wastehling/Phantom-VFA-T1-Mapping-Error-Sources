import torch


def T2star_coefficients_to_parameters(coefficients, parameter_ranges, mapping_method='sigmoid'):
    """
    converts T2star coefficients to parameters (T2star, S0)
    Args:
        coefficients: input coefficients
        parameter_ranges: ranges of output parameters to which input coefficients should be mapped
        mapping_method: method of mapping

    Returns:
        T2star, S0: T2star parameters
    """

    # retrieve parameter constraints
    T2starmin, T2starmax = parameter_ranges['T2star'][0], parameter_ranges['T2star'][1]
    S0min, S0max = parameter_ranges['S0'][0], parameter_ranges['S0'][1]

    # coefficient to parameter
    if mapping_method == 'sigmoid':
        T2star = T2starmin + torch.sigmoid(coefficients[:, 0].unsqueeze(1)) * (T2starmax - T2starmin)
        S0 = S0min + torch.sigmoid(coefficients[:, 1].unsqueeze(1)) * (S0max - S0min)
    elif mapping_method == 'linear':
        T2star = T2starmin + (coefficients[:, 0].unsqueeze(1) * (T2starmax - T2starmin))
        S0 = S0min + (coefficients[:, 1].unsqueeze(1) * (S0max - S0min))
    else:
        raise NotImplementedError(f'only linear and sigmoidal mapping method available, not {mapping_method}')

    return T2star, S0


def T2star_parameters_to_signal(T2star, S0, TE):
    """
    converts T2star parameters to predicted signal at specified TE
    Args:
        T2star: T2star
        S0: signal at TE=0
        TE: echo time

    Returns:
        signal: signal at specified TE
    """
    # Calculate signal based on estimated parameters
    signal = S0 * torch.exp(-TE / T2star)
    return signal