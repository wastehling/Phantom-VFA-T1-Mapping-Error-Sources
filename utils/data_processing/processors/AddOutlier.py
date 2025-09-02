import random
import numpy as np
import torch
import torchio
from torchio.transforms import Transform


class AddOutlier(Transform):
    """
    Add outlier to signals
    """

    def __init__(self, nr_outliers, outlier_fraction, probability, seed=None, **kwargs):
        """
        constructor for AddOutlier

        Args:
            nr_outliers: int, number of outliers to add
            outlier_fraction: float, fraction of signal to change data point with
            probability: probability of applying the transform
            seed: random seed
            **kwargs: kwargs for Transform
        """
        self.nr_outliers = nr_outliers
        self.outlier_fraction = outlier_fraction
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
        signals = self.add_outlier(signals, self.nr_outliers, self.outlier_fraction, self.rng)
        subject.add_image(torchio.Image(tensor=torch.Tensor(signals)), 'signals')
        return subject

    @staticmethod
    def add_outlier(signals, nr_outliers, outlier_fraction, rng):
        """
        add outlier
        Args:
             signals: signals array to add outlier to
             nr_outliers: number of outliers to add
             outlier_fraction: fraction of signal to change data point with
             rng: random number generator
         Returns:
             signals:  signals array

         """
        # normalize signals to S0 intensity
        outlier_idcs = rng.random_integers(0, signals.shape[0] - 1, nr_outliers)
        for outlier_idx in outlier_idcs:
            outlier_signal = signals[outlier_idx, :, :, :]
            if rng.choice([1, 2]) == 1:
                outlier_signal = outlier_signal - (outlier_fraction * outlier_signal)
            else:
                outlier_signal = outlier_signal + (outlier_fraction * outlier_signal)
            signals[outlier_idx, :, :, :] = outlier_signal

        return signals
