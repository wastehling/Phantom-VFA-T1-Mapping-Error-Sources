import numpy as np
import torch
import torchio
from torchio.transforms import Transform
from utils.data_processing.processors.SortSignalOnXval import SortSignalOnXval


class SubsetMeasurements(Transform):
    """
    Subset measurements
    """
    def __init__(self, remaining_measurements, **kwargs):
        """
        constructor for SubsetMeasurements
        Args:
            remaining_measurements: indices of remaining measurements
            **kwargs: Transform kwargs
        """
        self.remaining_measurements = remaining_measurements
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
        signals = signals[self.remaining_measurements, ...]
        xvals = np.squeeze(images_dict['xvals'].numpy())
        xvals = xvals[self.remaining_measurements, ...]
        subject.add_image(torchio.Image(tensor=torch.Tensor(signals)), 'signals')
        subject.add_image(torchio.Image(tensor=torch.Tensor(np.reshape(xvals, (xvals.shape[0], 1, 1, 1)))), 'xvals')
        return subject

