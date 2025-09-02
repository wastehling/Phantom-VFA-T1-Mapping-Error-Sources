import numpy as np
import torch
import torchio
import copy
from torchio.transforms import Transform


class TransformVFASignal(Transform):
    """
    Transform VFA signals to domain where signals start at M0 and decay to zero for ascending flip angles
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        """
        Transform VFA signals to domain where signals start at M0 and decay to zero for ascending flip angles
        Args:
            subject: Subject

        Returns:
            subject: Subject
        """
        images_dict = subject.get_images_dict()
        signals = images_dict['signals'].numpy()
        xvals = np.squeeze(images_dict['xvals'].numpy())
        signals = self.transform_signals(signals, xvals)
        subject.add_image(torchio.Image(tensor=torch.Tensor(signals)), 'signals')
        return subject

    @staticmethod
    def transform_signals(signals, xvals):
        """
        Transform VFA signals to domain where signals start at M0 and decay to zero for ascending flip angles
         Args:
             signals: signals array to transform
             xvals: xval array

         Returns:
             signals: transformed signals array

         """
        xvals = copy.deepcopy(xvals)
        signals /= np.sin(np.deg2rad(xvals))[:, np.newaxis, np.newaxis, np.newaxis]
        return signals


class MaxNormalizeVFASignal(Transform):
    """
    Normalize VFA signal based on max signal
    """
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
         normalize signals based on max signal
         Args:
             signals: signals array to normalize
             xvals: xval array

         Returns:
             normalized_signals: normalized signals array
             maxsignal: max signal

         """
        # get max signal along xval axis
        maxsignal = np.max(signals, axis=0)

        # normalize signals
        normalized_signals = signals / maxsignal

        return normalized_signals, maxsignal


