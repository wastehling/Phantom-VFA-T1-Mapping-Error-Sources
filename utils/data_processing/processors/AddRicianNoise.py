import random
import numpy as np
import torch
import torchio
from torchio.transforms import Transform


class AddRicianNoise(Transform):
    """
    Add Rician noise to signals
    """

    def __init__(self, sigma, probability, seed=None, **kwargs):
        """
        constructor for AddRicianNoise

        Args:
            sigma: float, standard deviation of the Gaussian noise elements
            probability: probability of applying the transform
            seed: random seed
            **kwargs: kwargs for Transform
        """
        self.sigma = sigma
        self.p = probability
        self.seed = seed if seed is not None else random.randint(0, 1000)
        self.rng = np.random.RandomState(self.seed)
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        """
        Args:
            subject: Subject

        Returns:
            subject: Subject
        """
        images_dict = subject.get_images_dict()
        signals = images_dict['signals'].numpy()
        signals = self.add_rician_noise(signals, self.sigma, self.rng)
        subject.add_image(torchio.Image(tensor=torch.Tensor(signals)), 'signals')
        return subject

    @staticmethod
    def add_rician_noise(signals, sigma, rng):
        """
        add Rician noise
        Args:
             signals: signals array to add Rician noise to
             sigma: standard deviation of the Gaussian noise elements
             rng: random number generator
         Returns:
             signals:  signals array

         """
        # create 2 signal arrays filled with gaussian noise
        noise_real = rng.normal(0, sigma, signals.shape)
        noise_imag = rng.normal(0, sigma, signals.shape)

        # add Rician noise to the simulated data
        signals = np.sqrt(np.power(signals + noise_real, 2) + np.power(noise_imag, 2)).squeeze()

        return signals
