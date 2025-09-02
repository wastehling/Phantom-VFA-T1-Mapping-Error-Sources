import torch


def vfa_coefficients_to_parameters(coefficients, parameter_ranges, mapping_method='sigmoid'):
    """
    converts vfa coefficients to parameters (M0, T1overTR)
    Args:
        coefficients: input coefficients
        parameter_ranges: ranges of output parameters to which input coefficients should be mapped
        mapping_method: method of mapping

    Returns:
        M0, T1overTR: vfa parameters
    """

    # retrieve parameter constraints
    M0min, M0max = parameter_ranges['M0'][0], parameter_ranges['M0'][1]
    T1min, T1max = parameter_ranges['T1'][0], parameter_ranges['T1'][1]
    TRmin, TRmax = parameter_ranges['TR'][0], parameter_ranges['TR'][1]
    T1overTRmin, T1overTRmax = T1min / TRmax, T1max / TRmin

    # coefficient to parameter
    if mapping_method == 'sigmoid':
        M0 = M0min + torch.sigmoid(coefficients[:, 0].unsqueeze(1)) * (M0max - M0min)
        T1overTR = T1overTRmin + torch.sigmoid(coefficients[:, 1].unsqueeze(1)) * (T1overTRmax - T1overTRmin)
    elif mapping_method == 'linear':
        M0 = M0min + (coefficients[:, 0].unsqueeze(1) * (M0max - M0min))
        T1overTR = T1overTRmin + (coefficients[:, 1].unsqueeze(1) * (T1overTRmax - T1overTRmin))
    else:
        raise NotImplementedError(f'only linear and sigmoidal mapping method available, not {mapping_method}')

    return M0, T1overTR


def vfa_parameters_to_signal(M0, T1, TR, FA):
    """
    simulate VFA signal (Mz), following:
    Mz(FA) = M0*sin(FA)*(1-e^(-TR/T1))/(1-cos(FA)e^(-TR/T1))

    Args:
        M0: tensor of magnetization at thermal equilibirum
        T1: tensor of T1
        TR: tensor of repetition time
        FA: tensor of flip angle

    Returns:
        simulated_data: simulated VFA signal

    """
    TRoverT1 = torch.div(TR, T1)
    relative_Mz = torch.sin(torch.deg2rad(FA)) * (1 - torch.exp(-TRoverT1)) / \
                  (1 - torch.cos(torch.deg2rad(FA)) * torch.exp(-TRoverT1))
    Mz = relative_Mz * M0
    return relative_Mz, Mz


def vfa_parameters_to_signal_transformed(M0, T1, TR, FA):
    """
    simulate VFA signal (Mz), following:
    Mz(FA) = M0*sin(FA)*(1-e^(-TR/T1))/(1-cos(FA)e^(-TR/T1))

    Args:
        M0: tensor of magnetization at thermal equilibirum
        T1: tensor of T1
        TR: tensor of repetition time
        FA: tensor of flip angle

    Returns:
        simulated_data: simulated VFA signal

    """
    TRoverT1 = torch.div(TR, T1)
    relative_Mz_over_sinFA = (1 - torch.exp(-TRoverT1)) / \
                  (1 - torch.cos(torch.deg2rad(FA)) * torch.exp(-TRoverT1))
    Mz_over_sinFA = relative_Mz_over_sinFA * M0
    return relative_Mz_over_sinFA, Mz_over_sinFA