import numpy as np
import torch
import torchio
import logging
import random
from torchio.transforms import Transform
from utils.data_processing.processors.SortSignalOnXval import SortSignalOnXval


class DropRandomMeasurements(Transform):
    """
    Drop random measurements
    """
    def __init__(self, nr_measurements, probability=1, seed_nr=1, qmri='VFA', s0_measurements=None,
                 potential_s0_measurements=None, dt_measurements=None, potential_dt_measurements=None, **kwargs):
        """
        constructor for DropRandomMeasurements
        Args:
            nr_measurements: number of measurements to keep
            probability: probability of applying the transform
            seed_nr: random seed
            **kwargs: Transform kwargs
        """
        self.nr_measurements = nr_measurements
        self.rng = np.random.default_rng(seed_nr)
        self.p = probability
        self.qmri = qmri
        self.s0_measurements = s0_measurements
        self.potential_s0_measurements = potential_s0_measurements
        self.dt_measurements = dt_measurements
        self.potential_dt_measurements = potential_dt_measurements
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
        xvals = np.squeeze(images_dict['xvals'].numpy())
        if isinstance(self.nr_measurements, list):
            subject_measurements = self.rng.choice(self.nr_measurements)
        else:
            subject_measurements = self.nr_measurements
        if self.qmri == 'VFA':
            signals, xvals = self.drop_random_measurements_vfa(signals, xvals, subject_measurements, rng=self.rng)
        elif self.qmri == 'IVIM':
            signals, xvals = self.drop_random_measurements_ivim(signals, xvals, subject_measurements, rng=self.rng)
        elif self.qmri == 'DCE':
            signals, xvals = self.drop_random_measurements_dce(signals, xvals, subject_measurements,
                                                               self.potential_s0_measurements, self.s0_measurements,
                                                               self.potential_dt_measurements, self.dt_measurements,
                                                               rng=self.rng)
        logging.info(f"Measurements remaining: {xvals}")
        subject.add_image(torchio.Image(tensor=torch.Tensor(signals)), 'signals')
        subject.add_image(torchio.Image(tensor=torch.Tensor(np.reshape(xvals, (xvals.shape[0], 1, 1, 1)))), 'xvals')
        return subject

    @staticmethod
    def drop_random_measurements_vfa(signals, xvals, remaining_measurements, rng):
        """
        drop random measurements

        Args:
            signals: signals
            xvals: corresponding xvals
            remaining_measurements: number of measurements to keep
            rng: random number generator

        Returns:
            signals: signals
            xvals: corresponding xvals

        """
        if xvals.shape[0] <= remaining_measurements:
            return signals, xvals

        if remaining_measurements < 2:
            return signals, xvals

        signals, xvals = SortSignalOnXval.sort_signals_on_xval_array(signals, xvals)

        # choose smallest flip angle
        smallest_flip_angle_idx = 0

        # choose flip angle in ernst angle regime
        bool_3over = xvals > 2
        bool_10under = xvals < 11
        bool_ernst = bool_3over & bool_10under
        ernst_angle_idcs = np.where(bool_ernst)[0]
        ernst_angle_idx = rng.choice(ernst_angle_idcs)

        # choose largest flip angle
        flip_angles_over15_idcs = np.where(xvals > 15)[0]
        if len(flip_angles_over15_idcs) > 0:
            largest_flip_angle_idx = rng.choice(flip_angles_over15_idcs)
        else:
            largest_flip_angle_idx = xvals.shape[0] - 1

        # choose remaining indices
        smallest_flip_angle = xvals[smallest_flip_angle_idx]
        largest_flip_angle = xvals[largest_flip_angle_idx]
        body_size = remaining_measurements - 3
        body = []
        for i in range(1, body_size+1):
            flip_angle_spread = largest_flip_angle - smallest_flip_angle
            flip_angle_to_approx = smallest_flip_angle + i * (flip_angle_spread / (body_size + 1))
            flip_angle_to_sample_idx = (np.abs(xvals - flip_angle_to_approx)).argmin()
            if flip_angle_to_sample_idx != ernst_angle_idx:
                body.append(flip_angle_to_sample_idx)
            else:
                body.append(flip_angle_to_sample_idx + 1)

        body_indices = np.asarray(body)
        remaining_indices = np.concatenate([np.array([smallest_flip_angle_idx]), np.array([ernst_angle_idx]),
                                            body_indices, np.array([largest_flip_angle_idx])]).astype('int')
        remaining_indices = np.sort(remaining_indices)
        remaining_xvals = xvals[remaining_indices, ...]
        remaining_signals = signals[remaining_indices, ...]
        remaining_xvals_unique, remaining_xvals_unique_idcs = np.unique(remaining_xvals, return_index=True)
        remaining_signals_unique = remaining_signals[remaining_xvals_unique_idcs]
        return remaining_signals_unique, remaining_xvals_unique.squeeze()

    @staticmethod
    def drop_random_measurements_ivim(signals, xvals, remaining_measurements, rng):
        """
        drop random measurements

        Args:
            signals: signals
            xvals: corresponding xvals
            remaining_measurements: number of measurements to keep
            rng: random number generator

        Returns:
            signals: signals
            xvals: corresponding xvals

        """
        if xvals.shape[0] <= remaining_measurements:
            return signals, xvals

        if remaining_measurements < 2:
            return signals, xvals

        signals, xvals = SortSignalOnXval.sort_signals_on_xval_array(signals, xvals)

        # choose smallest flip angle
        smallest_bval_idx = 0
        remaining_measurements -= 1

        # choose flip angle in ernst angle regime
        size_below_bval100  = int(np.ceil(remaining_measurements / 2))
        size_below_bval50 = int(np.ceil(size_below_bval100 / 2))
        size_between_bval_50_100 = size_below_bval100 - size_below_bval50
        size_above_bval100 = remaining_measurements - size_below_bval100

        bool_above_bval100 = xvals >= 100
        bool_below_bval50 = xvals < 50
        bool_between_bval_50_100 = np.logical_and(xvals >= 50, xvals < 100)

        bool_above_bval100_idcs = np.where(bool_above_bval100)[0]
        bool_below_bval50_idcs = np.where(bool_below_bval50)[0]
        bool_between_bval_50_100_idcs = np.where(bool_between_bval_50_100)[0]

        # random choose indices for each bval range
        if len(bool_above_bval100_idcs) > 0:
            above_bval_100_idcs = rng.choice(bool_above_bval100_idcs, size_above_bval100, replace=False)
        if len(bool_below_bval50_idcs) > 0:
            below_bval_50_idcs = rng.choice(bool_below_bval50_idcs, size_below_bval50, replace=False)
        if len(bool_between_bval_50_100_idcs) > 0:
            between_bval_50_100_idcs = rng.choice(bool_between_bval_50_100_idcs, size_between_bval_50_100, replace=False)

        # combine all bvals
        idcs = np.concatenate([np.array([smallest_bval_idx]), below_bval_50_idcs, between_bval_50_100_idcs, above_bval_100_idcs])
        remaining_indices = np.sort(idcs)
        remaining_xvals = xvals[remaining_indices, ...]
        remaining_signals = signals[remaining_indices, ...]
        remaining_xvals_unique, remaining_xvals_unique_idcs = np.unique(remaining_xvals, return_index=True)
        remaining_signals_unique = remaining_signals[remaining_xvals_unique_idcs]
        return remaining_signals_unique, remaining_xvals_unique.squeeze()

    @staticmethod
    def drop_random_measurements_dce(signals, xvals, remaining_measurements, potential_s0_measurements,
                                     s0_measurements, potential_dt_measurements, dt_measurements, rng):
        """
        drop random measurements

        Args:
            signals: signals
            xvals: corresponding xvals
            remaining_measurements: number of measurements to keep
            potential_s0_measurements: number of measurements before approximate dt that can be sampled
            s0_measurements: number of measurements before approximate dt
            potential_dt_measurements: number of measurements after potential s0 measurements that can be sampled
            dt_measurements: number of measurements around approximate_dt
            rng: random number generator

        Returns:
            signals: signals
            xvals: corresponding xvals

        """

        if xvals.shape[0] <= remaining_measurements:
            return signals, xvals

        if isinstance(potential_dt_measurements, list):
            potential_dt_measurements = rng.choice(potential_dt_measurements)

        if isinstance(dt_measurements, list):
            dt_measurements = rng.choice(dt_measurements)

        if isinstance(potential_s0_measurements, list):
            potential_s0_measurements = rng.choice(potential_s0_measurements)

        if isinstance(s0_measurements, list):
            s0_measurements = rng.choice(s0_measurements)

        signals, xvals = SortSignalOnXval.sort_signals_on_xval_array(signals, xvals)

        # determine amount of measurements per phase
        delayed_phase_measurements = remaining_measurements - s0_measurements - dt_measurements

        # equidistantly sample indices for s0 phase
        bool_potential_s0_measurements = np.zeros_like(xvals)
        bool_potential_s0_measurements[:potential_s0_measurements] = True
        potential_s0_indices = [i for i, val in enumerate(bool_potential_s0_measurements) if val]
        s0_measurements_idcs = sorted(random.sample(potential_s0_indices, s0_measurements))

        # equidistantly sample indices for dt phase
        bool_potential_dt_measurements = np.zeros_like(xvals)
        bool_potential_dt_measurements[potential_s0_measurements:potential_s0_measurements + potential_dt_measurements] = True
        idcs_potential_dt_measurements = np.argwhere(bool_potential_dt_measurements).flatten()
        sampled_dt_indices = np.linspace(0, len(idcs_potential_dt_measurements) - 1, dt_measurements,
                                      dtype=int)
        dt_measurements_idcs = idcs_potential_dt_measurements[sampled_dt_indices]

        # equidistantly sample indices for delayed phase
        bool_potential_delayed_phase_measurements = np.zeros_like(xvals)
        bool_potential_delayed_phase_measurements[potential_s0_measurements + potential_dt_measurements:] = True
        idcs_bool_delayed_phase_measurements = np.argwhere(bool_potential_delayed_phase_measurements).flatten()
        sampled_indices = np.linspace(0, len(idcs_bool_delayed_phase_measurements) - 1, delayed_phase_measurements,
                                      dtype=int)
        delayed_phase_measurements_idcs = idcs_bool_delayed_phase_measurements[sampled_indices]

        # combine all indices
        idcs = np.concatenate([s0_measurements_idcs, dt_measurements_idcs, delayed_phase_measurements_idcs])
        remaining_indices = np.sort(idcs)
        remaining_xvals = xvals[remaining_indices, ...]
        remaining_signals = signals[remaining_indices, ...]
        remaining_xvals_unique, remaining_xvals_unique_idcs = np.unique(remaining_xvals, return_index=True)
        remaining_signals_unique = remaining_signals[remaining_xvals_unique_idcs]
        return remaining_signals_unique, remaining_xvals_unique.squeeze()