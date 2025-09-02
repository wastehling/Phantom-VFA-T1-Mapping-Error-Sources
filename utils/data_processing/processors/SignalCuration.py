import numpy as np
import torch
import torchio
from torchio.transforms import Transform


class SignalCuration(Transform):
    """
    Curate signals based on qmri application
    """
    def __init__(self, qmri_application, **kwargs):
        self.qmri_application = qmri_application
        super().__init__(**kwargs)

    def apply_transform(self, subject):
        """
        curates signals Image of subject
        Args:
            subject: Subject

        Returns:
            subject: Subject
        """
        images_dict = subject.get_images_dict()
        if self.qmri_application == 'IVIM':
            signals = images_dict['signals'].numpy()
            xvals = np.squeeze(images_dict['xvals'].numpy())
            valid_mask = self.ivim_selection(signals, xvals)
        elif self.qmri_application == 'VFA':
            signals = images_dict['signals'].numpy()
            valid_mask = self.vfa_selection(signals)
        else:
            raise NotImplementedError

        if 'valid_mask' in images_dict.keys():
            existing_valid_mask = subject.get('valid_mask').cpu().numpy()
            valid_mask = np.logical_and(valid_mask, existing_valid_mask)

        try:
            subject.add_image(torchio.Image(tensor=torch.Tensor(valid_mask)), 'valid_mask')
        except ValueError:
            subject.add_image(torchio.Image(tensor=torch.Tensor(np.expand_dims(valid_mask, 0))), 'valid_mask')
        return subject

    @staticmethod
    def ivim_selection(signals, xvals):
        """
        returns only those signals exhibiting ivim decay
        Args:
            signals: signals for corresponding xvals
            xvals: xvals

        Returns:
            mask_signals: mask for signals
        """

        # get average b0 signals and set signals with S0 of 0 to nan
        mean_S0 = np.nanmean(signals[xvals <= 0.0001, ...], axis=0)

        # select only those voxels with average S0 larger than half of median S0 of voxels with S0 larger than 0
        valid_idcs_median_value = mean_S0 > (0.5 * np.nanmedian(mean_S0[mean_S0 > 0]))

        # check if signal is ivim like
        try:
            signals[signals < 0] = 0
            signals[signals > 1.5] = 1.5
            valid_idcs_ivim_curve1 = np.percentile(signals[xvals < 50, ...], 95,
                                                   axis=0) < 1.3
            valid_idcs_ivim_curve2 = np.percentile(signals[xvals > 50, ...], 95,
                                                   axis=0) < 1.2
            valid_idcs_ivim_curve3 = np.percentile(signals[xvals > 150, ...], 95,
                                                   axis=0) < 1.0
            mask_signals = valid_idcs_median_value & valid_idcs_ivim_curve1 & valid_idcs_ivim_curve2 & valid_idcs_ivim_curve3
        except IndexError:
            mask_signals = valid_idcs_median_value
        return mask_signals

    @staticmethod
    def vfa_selection(signals):
        """
        returns only those signals exhibiting potential vfa trajectory
        Args:
            signals: signals

        Returns:
            mask
        """

        mask = np.min(signals, axis=0) > 0
        mask2 = signals[0, ...] > 0
        mask_total = mask & mask2
        return mask_total
