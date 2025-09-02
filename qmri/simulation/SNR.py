import numpy as np


def create_array_of_SNR_values(SNR, samples, random_generator=None):
    """
    creates array of SNR values
    Args:
        SNR: bounds of SNR
        samples: amount of SNR values required
        random_generator: random generator or None

    Returns:
        SNR_array: array of SNR values

    """
    if random_generator is None:
        random_generator = np.random.RandomState()
    test = random_generator.uniform(0, 1, (samples, 1))
    SNR_array = np.exp(np.log(SNR[1]) + (test * (np.log(SNR[0]) - np.log(SNR[1]))))
    return SNR_array
