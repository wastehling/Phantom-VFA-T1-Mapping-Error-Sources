"""
The following functions are adapted from IVIMNET (https://github.com/oliverchampion/IVIMNET):
- fit_least_squares_array_ivim()
- fit_least_squares_ivim()
- fit_segmented()
- order()
- ivimN()
- ivimN_noS0()
- ivim()
Copyright 2021 Oliver Gurney-Champion
Licensed under the GNU General Public License v3.0
"""
import numpy as np
import logging

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import curve_fit


def fit_least_squares_array_ivim(bvalues, dw_data, S0_output=True, fitS0=True, njobs=4,
                                 bounds=([0, 0, 0.005, 0.7], [0.005, 0.7, 0.2, 1.3])):
    """
    This is an implementation of the conventional IVIM fit. It is fitted in array form.
    :param bvalues: 1D Array with the b-values
    :param dw_data: 2D Array with diffusion-weighted signal in different voxels at different b-values
    :param S0_output: Boolean determining whether to output (often a dummy) variable S0; default = True
    :param fitS0: Boolean determining whether to fix S0 to 1; default = False
    :param njobs: Integer determining the number of parallel processes; default = 4
    :param bounds: Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]). Default: ([0.005, 0, 0, 0.8], [0.2, 0.7, 0.005, 1.2])
    :return Dt: 1D Array with D in each voxel
    :return Fp: 1D Array with f in each voxel
    :return Dp: 1D Array with Dp in each voxel
    :return S0: 1D Array with S0 in each voxel
    """
    # normalise the data to S(value=0)
    S0 = np.mean(dw_data[:, bvalues <= 0.01], axis=1)
    dw_data = dw_data / S0[:, None]
    single = False

    # check if parallel is desired
    if njobs > 1:
        try:
            # defining parallel function
            def parfun(i):
                return fit_least_squares_ivim(bvalues, dw_data[i, :], S0_output=S0_output, fitS0=fitS0, bounds=bounds)

            output = Parallel(n_jobs=njobs)(delayed(parfun)(i) for i in tqdm(range(len(dw_data)), position=0, leave=True))
            Dt, Fp, Dp, S0 = np.transpose(output)
        except:
            single = True
    else:
        single = True
    if single:
        # run on single core, instead. Defining empty arrays
        Dp = np.zeros(len(dw_data))
        Dt = np.zeros(len(dw_data))
        Fp = np.zeros(len(dw_data))
        S0 = np.zeros(len(dw_data))
        # running in a single loop and filling arrays
        for i in tqdm(range(len(dw_data)), position=0, leave=True):
            Dt[i], Fp[i], Dp[i], S0[i] = fit_least_squares_ivim(bvalues, dw_data[i, :], S0_output=S0_output, fitS0=fitS0,
                                                                bounds=bounds)
    return [Dt, Fp, Dp, S0]


def fit_least_squares_ivim(bvalues, dw_data, S0_output=False, fitS0=True,
                           bounds=([0, 0, 0.005, 0.7],[0.005, 0.7, 0.2, 1.3])):
    """
    This is an implementation of the conventional IVIM fit. It fits a single curve
    :param bvalues: Array with the b-values
    :param dw_data: Array with diffusion-weighted signal at different b-values
    :param S0_output: Boolean determining whether to output (often a dummy) variable S0; default = True
    :param fitS0: Boolean determining whether to fix S0 to 1; default = False
    :param bounds: Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]). Default: ([0.005, 0, 0, 0.8], [0.2, 0.7, 0.005, 1.2])
    :return Dt: Array with D in each voxel
    :return Fp: Array with f in each voxel
    :return Dp: Array with Dp in each voxel
    :return S0: Array with S0 in each voxel
    """

    # bounds are rescaled such that each parameter changes at roughly the same rate to help fitting.
    bounds = ([bounds[0][0] * 1000, bounds[0][1] * 10, bounds[0][2] * 10, bounds[0][3]],
              [bounds[1][0] * 1000, bounds[1][1] * 10, bounds[1][2] * 10, bounds[1][3]])
    params, _ = curve_fit(ivimN, bvalues, dw_data, p0=[1, 1, 1, 1], bounds=bounds)
    S0 = params[3]

    # correct for the rescaling of parameters
    Dt, Fp, Dp = params[0] / 1000, params[1] / 10, params[2] / 10

    # reorder output in case Dp<Dt
    return order(Dt, Fp, Dp, S0)


def fit_least_squares_array_vfa(xvalues, signal, njobs, bounds, TR):
    """
    This is an implementation of the conventional VFA fit. It is fitted in array form.
    :param xvalues: 1D Array with the flip angles
    :param signal: 2D Array with signal at different flip angles
    :param njobs: Integer determining the number of parallel processes
    :param bounds: Array with fit bounds ([M0min, T1min],[M0max, T1max])
    :param TR: Repetition time
    :return M0: 1D Array with M0 in each voxel
    :return T1: 1D Array with T1 in each voxel
    """
    try:
        def parfun(i):
            return fit_least_squares_vfa(xvalues, signal[i, :], bounds, TR[i])

        output = Parallel(n_jobs=njobs)(delayed(parfun)(i) for i in tqdm(range(len(signal)), position=0, leave=True))
        M0, T1 = np.transpose(output)
    except:
        M0 = np.zeros(len(signal))
        T1 = np.zeros(len(signal))
        for i in range(len(signal)):
            try:
                M0[i], T1[i] = fit_least_squares_vfa(xvalues, signal[i, :], bounds, TR[i])
            except IndexError as e:
                M0[i], T1[i] = fit_least_squares_vfa(xvalues, signal[i, :], bounds, TR)
    return [M0, T1]


def fit_least_squares_vfa(xvalues, dw_data, bounds, TR):
    """
    This is an implementation of the conventional VFA fit. It fits a single curve
    :param xvalues: 1D Array with the flip angles
    :param dw_data: 1D Array with signal at different flip angles
    :param bounds: Array with fit bounds ([M0min, T1min],[M0max, T1max])
    :param TR: Repetition time
    """
    VFA_minimizer = VFA(TR)

    # set bounds
    bounds = ([bounds[0][0], bounds[0][1]],
              [bounds[1][0], bounds[1][1]])
    params, _ = curve_fit(VFA_minimizer.vfa, xvalues, dw_data, p0=[1, 1], bounds=bounds)

    # unpack parameters
    M0, T1 = params[0], params[1]

    return M0, T1

def fit_segmented(bvalues, dw_data, bounds=([0, 0, 0.005],[0.005, 0.7, 0.2]), cutoff=75):
    """
    This is an implementation of the segmented fit, in which we first estimate D using a curve fit to b-values>cutoff;
    then estimate f from the fitted S0 and the measured S0 and finally estimate D* while fixing D and f.
    :param bvalues: Array with the b-values
    :param dw_data: Array with diffusion-weighted signal at different b-values
    :param bounds: Array with fit bounds ([Dtmin, Fpmin, Dpmin, S0min],[Dtmax, Fpmax, Dpmax, S0max]). Default: ([0.005, 0, 0, 0.8], [0.2, 0.7, 0.005, 1.2])
    :param cutoff: cutoff value for determining which data is taken along in fitting D
    :return Dt: Fitted D
    :return Fp: Fitted f
    :return Dp: Fitted Dp
    :return S0: Fitted S0
    """
    try:
        # determine high b-values and data for D
        high_b = bvalues[bvalues >= cutoff]
        high_dw_data = dw_data[bvalues >= cutoff]
        # correct the bounds. Note that S0 bounds determine the max and min of f
        bounds1 = ([bounds[0][0] * 1000., 1 - bounds[1][1]], [bounds[1][0] * 1000., 1. - bounds[0][
            1]])  # By bounding S0 like this, we effectively insert the boundaries of f
        # fit for S0' and D
        params, _ = curve_fit(lambda b, Dt, int: int * np.exp(-b * Dt / 1000), high_b, high_dw_data, p0=(1, 1),
                              bounds=bounds1)
        Dt, Fp = params[0] / 1000, 1 - params[1]
        # remove the diffusion part to only keep the pseudo-diffusion
        dw_data_remaining = dw_data - (1 - Fp) * np.exp(-bvalues * Dt)
        bounds2 = (bounds[0][2], bounds[1][2])
        # fit for D*
        params, _ = curve_fit(lambda b, Dp: Fp * np.exp(-b * Dp), bvalues, dw_data_remaining, p0=(0.1), bounds=bounds2)
        Dp = params[0]
        return Dt, Fp, Dp
    except:
        # if fit fails, return zeros
        # print('segmented fit failed')
        return 0., 0., 0.


def order(Dt, Fp, Dp, S0=None):
    """
    Function to reorder D* and D in case they were swapped during unconstraint fitting. Forces D* > D (Dp>Dt)
    Args:
        Dt: pseudo-diffusion coefficient
        Fp: perfusion fraction
        Dp: diffusion coefficient
        S0: Signal at B0

    Returns:
        Dt: pseudo-diffusion coefficient
        Fp: perfusion fraction
        Dp: diffusion coefficient
        S0: Signal at B0

    """
    if Dp < Dt:
        Dp, Dt = Dt, Dp
        Fp = 1 - Fp
    if S0 is None:
        return Dt, Fp, Dp
    else:
        return Dt, Fp, Dp, S0


def ivimN(bvalues, Dt, Fp, Dp, S0):
    """
    IVIM function in which we try to have equal variance in the different IVIM parameters
    Args:
        bvalues: 1D Array with the b-values
        Dt: 1D Array with D in each voxel
        Fp: 1D Array with f in each voxel
        Dp: 1D Array with Dp in each voxel
        S0: 1D Array with S0 in each voxel

    Returns:
        1D Array with ivim signal
    """
    return S0 * ivimN_noS0(bvalues, Dt, Fp, Dp)


def ivimN_noS0(bvalues, Dt, Fp, Dp):
    """
    IVIM function in which we try to have equal variance in the different IVIM parameters and S0=1
    Args:
        bvalues: 1D Array with the b-values
        Dt: 1D Array with D in each voxel
        Fp: 1D Array with f in each voxel
        Dp: 1D Array with Dp in each voxel

    Returns:
        1D Array of ivim signal
    """
    return (Fp / 10 * np.exp(-bvalues * Dp / 10) + (1 - Fp / 10) * np.exp(-bvalues * Dt / 1000))


def ivim(bvalues, Dt, Fp, Dp, S0):
    """
    IVIM function
    Args:
        bvalues: 1D Array with the b-values
        Dt: 1D Array with D in each voxel
        Fp: 1D Array with f in each voxel
        Dp: 1D Array with Dp in each voxel
        S0: 1D Array with S0 in each voxel

    Returns:
        1D Array of ivim signal
    """
    return (S0 * (Fp * np.exp(-bvalues * Dp) + (1 - Fp) * np.exp(-bvalues * Dt)))


class VFA:
    def __init__(self, TR):
        """
        VFA signal equation (Mz)

        Args:
            TR: array of repetition times

        Returns:
            VFA signal

        """
        self.TR = TR
    def vfa(self, FA, M0, T1):
        """
        VFA signal equation (Mz)
        Args:
            FA: array of flip angles
            M0: array of magnetization at thermal equilibirum
            T1: array of T1

        Returns:
            Mz: VFA signal

        """
        TRoverT1 = self.TR/T1
        relative_Mz = np.sin(np.deg2rad(FA)) * (1 - np.exp(-TRoverT1)) / \
                      (1 - np.cos(np.deg2rad(FA)) * np.exp(-TRoverT1))
        Mz = relative_Mz * M0
        return Mz

def fit_least_squares_array_t2star(xvalues, signal, njobs, bounds):
    """
    This is an implementation of the conventional T2star fit. It is fitted in array form.
    Args:
        xvalues: 1D Array with the echo times
        signal: 2D Array with signal at different echo times
        njobs: Integer determining the number of parallel processes
        bounds: Array with fit bounds ([T2starmin, S0min],[T2starmax, S0max])
    Returns:
        T2star: 1D Array with T2star in each voxel
        S0: 1D Array with S0 in each voxel
    """
    try:
        # defining parallel function
        def parfun(i):
            return fit_least_squares_t2star(xvalues, signal[i, :], bounds)

        # run parallel
        output = Parallel(n_jobs=njobs)(delayed(parfun)(i) for i in tqdm(range(len(signal)), position=0, leave=True))
        T2star, S0 = np.transpose(output)
    except:
        T2star = np.zeros(len(signal))
        S0 = np.zeros(len(signal))
        # run on single core, instead
        for i in range(len(signal)):
            T2star[i], S0[i] = fit_least_squares_t2star(xvalues, signal[i, :], bounds)
    return [T2star, S0]


def fit_least_squares_t2star(xvalues, dw_data, bounds):
    """
    This is an implementation of the conventional T2star fit. It fits a single curve
    """

    # set bounds
    bounds = ([bounds[0][0], bounds[0][1]],
              [bounds[1][0], bounds[1][1]])
    params, _ = curve_fit(t2star, xvalues, dw_data, p0=[0.03, 1], bounds=bounds, maxfev=100000)

    # unpack parameters
    T2star, S0 = params[0], params[1]

    return T2star, S0


def t2star(xvalues, T2star, S0):
    """
    T2star signal equation
    Args:
        xvalues: 1D Array with the echo times
        T2star: 1D Array with T2star in each voxel
        S0: 1D Array with S0 in each voxel

    Returns:
        1D Array of T2star signal
    """
    return S0 * np.exp(-xvalues / T2star)