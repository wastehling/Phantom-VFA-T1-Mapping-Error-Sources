import numpy as np
import torch
import torchio
import copy
from torchio.transforms import Transform


class LFANormalizeVFASignal(Transform):
    """
    Normalize signal array by dividing by the low flip angle signal. Based on the assumption that the signal at low flip
    angles is an approximation of M0 * sin(FA(rad)).
    """

    def __init__(self, lower_limit, upper_limit, **kwargs):
        """
        constructor for LFANormalizeVFASignal
        Args:
            lower_limit: lower limit for M0_approx
            upper_limit: upper limit for M0_approx
            **kwargs:
        """
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        """
        normalize VFA signals Image of subject
        Args:
            subject: Subject

        Returns:
            subject: Subject
        """
        images_dict = subject.get_images_dict()
        signals = images_dict['signals'].numpy()
        xvals = np.squeeze(images_dict['xvals'].numpy())
        signals, M0_approx = self.normalize_signals(signals, xvals)
        subject.add_image(torchio.Image(tensor=torch.Tensor(signals)), 'signals')
        try:
            M0s = images_dict['M0_image'].numpy()
            subject.add_image(torchio.Image(tensor=torch.Tensor(M0s/M0_approx)), 'M0_image')
        except:
            pass
        return subject

    def normalize_signals(self, signals, xvals):
        """
         normalize signals based on low flip angle M0 approximation
         Args:
             signals: signals array to normalize
             xvals: xval array

         Returns:
             normalized_signals: normalized signals array
            M0_approx: M0 approximation

         """
        # low flip angle approximation
        xvals = copy.deepcopy(xvals)
        minFA = np.min(xvals[np.nonzero(xvals)])
        minFA_idx = np.where(xvals == minFA)[0][0]
        minFA_rad = np.deg2rad(minFA)
        M0_approx = signals[minFA_idx]/np.sin(minFA_rad)

        M0_approx[M0_approx < self.lower_limit] = self.lower_limit
        M0_approx[M0_approx > self.upper_limit] = self.upper_limit

        # normalize signals to M0 approx
        try:
            normalized_signals = signals / M0_approx
        except:
            FloatingPointError('M0_approx is 0, cannot normalize signals')
            normalized_signals = signals

        return normalized_signals, M0_approx


class MaxNormalizeVFASignal(Transform):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        """
        normalize VFA signals Image of subject
        Args:
            subject: Subject

        Returns:
            subject: Subject
        """
        images_dict = subject.get_images_dict()
        signals = images_dict['signals'].numpy()
        xvals = np.squeeze(images_dict['xvals'].numpy())
        M0s = np.squeeze(images_dict['M0_image'].numpy())
        signals, maxsignal = self.normalize_signals(signals, xvals)
        subject.add_image(torchio.Image(tensor=torch.Tensor(signals)), 'signals')
        subject.add_image(torchio.Image(tensor=torch.Tensor(np.expand_dims(M0s/maxsignal, 0))), 'M0_image')
        return subject

    @staticmethod
    def normalize_signals(signals, xvals):
        """
         normalize signals
         Args:
             signals: signals array to normalize
             xvals: xval array

         Returns:
             normalized_signals: normalized signals array

         """
        # get initial slope of curve
        maxsignal = np.max(signals, axis=0)

        # normalize signals to dydx
        normalized_signals = signals / maxsignal

        return normalized_signals, maxsignal


