import numpy as np
import torch
import torchio

from torchio.transforms import Transform


class FlattenImageData(Transform):
    """
    Flatten image data
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        """
        flattens image data of signals image of subject
        Args:
            subject: Subject
        Returns:
            subject: Subject
        """
        images_dict = subject.get_images_dict(include=self.include, exclude=self.exclude)
        for image_key, image in images_dict.items():
            flattened_array = self.flatten_image_data(image.numpy())
            subject.add_image(torchio.Image(tensor=torch.Tensor(np.reshape(flattened_array,
                                                                           (flattened_array.shape[0],
                                                                            flattened_array.shape[1], 1, 1)))),
                              image_key)
        return subject

    @staticmethod
    def flatten_image_data(signals):
        """
        Flattens 4D array into 2D array
        Args:
            signals: signals array to flatten

        Returns:
            signals_array: 2D array
        """
        bvals, x, y, z = signals.shape
        signals_array = np.reshape(signals, (bvals, x * y * z))
        return signals_array

