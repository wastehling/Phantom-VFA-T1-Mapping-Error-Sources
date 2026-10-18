import os
import re

import numpy as np
import pandas as pd

REFERENCE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_values')


def _normalize_dashes(text):
    return re.sub(r'[‐‑‒–—―−]', '-', str(text))


def _interpolate_T1(temperatures, T1s, temperature):
    """Linear interpolation (and extrapolation) of the NIST T1 measurements to the phantom temperature."""
    order = np.argsort(temperatures)
    temperatures, T1s = np.asarray(temperatures, float)[order], np.asarray(T1s, float)[order]
    if temperature <= temperatures[0]:
        idx = 0
    elif temperature >= temperatures[-1]:
        idx = len(temperatures) - 2
    else:
        idx = np.searchsorted(temperatures, temperature) - 1
    slope = (T1s[idx + 1] - T1s[idx]) / (temperatures[idx + 1] - temperatures[idx])
    return int(T1s[idx] + slope * (temperature - temperatures[idx]))


def reference_T1(phantom, temperature):
    """
    Temperature-corrected reference T1 values in ms, sorted ascending.
    Args:
        phantom: 'GSP' (Gold Standard Phantoms MultiSample 120S) or 'CM' (CaliberMRI system phantom, NiCl2 layer)
        temperature: phantom temperature in degrees Celsius

    Returns:
        array of reference T1 values
    """
    if phantom == 'GSP':
        vials = [_normalize_dashes(v) for v in
                 pd.read_csv(os.path.join(REFERENCE_FOLDER, 'gs_vial_names.txt'), header=None)[0]]
        df = pd.read_excel(os.path.join(REFERENCE_FOLDER, 'O-41770_GoldStandPhant_GSP_T1T2_20230125.xlsx'),
                           engine='openpyxl')
        df['Order code name'] = df['Order code name'].map(_normalize_dashes)
        T1s = [_interpolate_T1(df.loc[df['Order code name'] == v, 'Temperature'],
                               df.loc[df['Order code name'] == v, 'T1 reported'], temperature) for v in vials]
    elif phantom == 'CM':
        df = pd.read_excel(os.path.join(REFERENCE_FOLDER, 'reference_measurements_130-0062.xlsx'), engine='openpyxl',
                           sheet_name='NiCl Solutions @ 3.0T', skiprows=3, skipfooter=5)
        T1s = [_interpolate_T1(df.loc[df['Solution'] == s, 'Temperature (C)'], df.loc[df['Solution'] == s, 'T1 (ms)'],
                               temperature) for s in df['Solution'].unique()]
    else:
        raise ValueError(f'Unknown phantom {phantom}')
    return np.sort(T1s)
