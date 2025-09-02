import torch
import torchio
import numpy as np

from torchio.transforms import Transform
from qmri.forward_model.dce import dce_to_r1eff_OGC, r1eff_to_conc


class SignalToConc(Transform):
    """
    Convert signals to concentration
    """
    def __init__(self, TR, flip_angle, relaxivity, R1=None, S0_timepoints=2, **kwargs):
        """
        constructor for SignalToConc
        Args:
            TR: repetition time
            flip_angle: flip angle
            relaxivity: Relaxivity
            R1: R1 map
            S0_timepoints: number of timepoints to average for S0
            **kwargs:
        """
        self.S0_timepoints = S0_timepoints
        self.TR = TR
        self.R1 = R1
        self.flip_angle = flip_angle
        self.relaxivity = relaxivity
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
        try:
            t1_map = images_dict['T1_image'].numpy()
            conc = self.signals_to_conc_withT1map(signals, t1_map)
        except KeyError:
            conc = self.signals_to_conc_withoutT1map(signals)
        subject.add_image(torchio.Image(tensor=torch.Tensor(conc)), 'signals')
        return subject

    def signals_to_conc_withT1map(self, signals, t1_map):
        """
        Convert signals to concentration using T1 map
         Args:
             signals: signals array
             t1_map: T1 map

         Returns:
             conc: concentration array

         """
        S0 = np.average(signals[0:self.S0_timepoints, ...], axis=0)
        t1_map[t1_map == 0] = 'nan'
        r1_map = 1 / (t1_map/1000)
        r1eff = dce_to_r1eff_OGC(signals, S0, r1_map, self.TR, self.flip_angle)
        conc = r1eff_to_conc(r1eff, r1_map, self.relaxivity)
        return conc

    def signals_to_conc_withoutT1map(self, signals):
        """
        Convert signals to concentration without T1 map
         Args:
             signals: signals array to convert to concentration

         Returns:
             conc: concentration array

         """
        S0 = np.average(signals[0:self.S0_timepoints, ...], axis=0)
        r1eff = dce_to_r1eff_OGC(signals, S0, self.R1, self.TR, self.flip_angle)
        conc = r1eff_to_conc(r1eff, self.R1, self.relaxivity)
        return conc


