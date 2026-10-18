"""
Step 1 of reproducing the VFA phantom paper: voxel-wise VFA T1 fitting in automatically generated vial masks.

For every study in the parameter file, the VFA series are loaded from the DICOM export, the B1+ map is resampled onto the
VFA grid, circular vial masks are generated on the configured slices, and T1 is fitted voxel-wise with the SPGR signal
equation (actual flip angle = nominal flip angle x B1+). Vial means are compared with temperature-corrected reference
values and written to <output_folder>/fitted_values_T1.csv. Masks (NIfTI) and mask QC images are stored as well.
"""
import argparse
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from models.classical_fitting.LSQ.least_squares import fit_least_squares_vfa
from qmri.data_loading.vfa.load_vfa_philips_dicom import index_session, load_vfa_series, load_b1_map, \
    resample_to_slice
from scripts.vfa_phantom_paper.reference_values import reference_T1
from scripts.vfa_phantom_paper.vial_masks import detect_vials, circular_vial_masks


def find_session(data_root, patient_name):
    """Returns the export folder of an exam; folders are named after the patient name (e.g. M_P5_20260518_...)."""
    hits = [name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name)) and (name == patient_name or name.startswith(patient_name + '_'))]
    if len(hits) != 1:
        raise ValueError(f'Expected one folder for {patient_name} in {data_root}, found {hits}')
    return index_session(os.path.join(data_root, hits[0]))


def fit_voxels(flip_angles, signals, TR, b1, bounds):
    """
    Voxel-wise VFA fit with B1+ corrected flip angles.
    Args:
        flip_angles: nominal flip angles [n_fa] in degrees
        signals: [n_voxels, n_fa]
        TR: repetition time in ms
        b1: B1+ scaling factor per voxel [n_voxels], or None
        bounds: ([M0min, T1min], [M0max, T1max])

    Returns:
        T1 per voxel, nan if the fit failed
    """
    T1 = np.full(len(signals), np.nan)
    for idx, signal in enumerate(signals):
        actual_flip_angles = flip_angles * (1.0 if b1 is None else b1[idx])
        p0 = (min(10 * signal.max(), 0.9 * bounds[1][0]), 1000)
        try:
            T1[idx] = fit_least_squares_vfa(actual_flip_angles, signal, bounds, TR, p0=p0)[1]
        except RuntimeError:
            pass
    return T1


def make_vial_masks(signals, phantom_cfg):
    """Detects the vials on the lowest flip angle and returns a 3D label image on the configured slices."""
    labels = np.zeros(signals.shape[1:], dtype=np.int16)
    central_slices = phantom_cfg['mask_slices']
    detection_image = signals[0][central_slices].mean(axis=0)
    vials = detect_vials(detection_image, phantom_cfg['n_vials'])
    vial_labels = circular_vial_masks(detection_image.shape, vials, phantom_cfg['mask_shrink_px'])
    for slice_index in central_slices:
        labels[slice_index] = vial_labels
    return labels, detection_image


def save_mask(labels, affine, path):
    """Saves a label image [slice, row, column] with an LPS affine as NIfTI (RAS)."""
    affine_ras = np.diag([-1, -1, 1, 1]) @ affine
    nib.save(nib.Nifti1Image(np.transpose(labels, (2, 1, 0)), affine_ras), path)


def save_mask_qc(image, labels, b1, title, path):
    fig, ax = plt.subplots(1, 2 if b1 is not None else 1, figsize=(12 if b1 is not None else 6, 6), squeeze=False)
    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].contour(labels > 0, levels=[0.5], colors='r', linewidths=0.6)
    for idx in range(1, labels.max() + 1):
        rows, cols = np.nonzero(labels == idx)
        ax[0, 0].text(cols.mean(), rows.mean(), str(idx), color='yellow', ha='center', va='center', fontsize=7)
    ax[0, 0].set_title(title)
    if b1 is not None:
        im = ax[0, 1].imshow(b1, vmin=0.5, vmax=1.5, cmap='gray')
        ax[0, 1].contour(labels > 0, levels=[0.5], colors='r', linewidths=0.6)
        ax[0, 1].set_title('B1+ (resampled)')
        fig.colorbar(im, ax=ax[0, 1], fraction=0.046)
    for a in ax.flat:
        a.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fit_study(phantom, study, study_cfg, vfa, masks, detection_image, name, session_index, parameters):
    """Fits all vials of one study, saves a mask QC image and returns one record per vial (sorted by T1)."""
    phantom_cfg = parameters['phantoms'][phantom]
    fit_cfg = parameters['fit']
    flip_angles, signals, affine, TR = vfa
    bounds = ([fit_cfg['parameter_ranges']['M0'][0], fit_cfg['parameter_ranges']['T1'][0]],
              [fit_cfg['parameter_ranges']['M0'][1], fit_cfg['parameter_ranges']['T1'][1]])

    # B1+ map resampled onto every mask slice
    b1_slices = {slice_index: None for slice_index in phantom_cfg['mask_slices']}
    if study_cfg['b1'] is not None:
        b1_volume = load_b1_map(session_index[study_cfg['b1']])
        for slice_index in b1_slices:
            b1_slices[slice_index] = resample_to_slice(*b1_volume, affine, signals.shape[2:], slice_index)

    first_slice = phantom_cfg['mask_slices'][0]
    save_mask_qc(detection_image, masks[first_slice], b1_slices[first_slice], f'{phantom} {study} ({name})',
                 os.path.join(parameters['output_folder'], 'masks', f'{phantom}_{study}_qc.png'))

    voxel_T1 = {idx: [] for idx in range(1, masks.max() + 1)}
    for slice_index, b1 in b1_slices.items():
        if b1 is not None:
            b1 = np.clip(b1, *fit_cfg['b1_clip'])
        for idx in voxel_T1:
            in_vial = masks[slice_index] == idx
            voxel_T1[idx].append(fit_voxels(flip_angles, signals[:, slice_index][:, in_vial].T, TR,
                                            None if b1 is None else b1[in_vial], bounds))

    measured = np.sort([int(np.nanmean(np.concatenate(values))) for values in voxel_T1.values()])
    reference = reference_T1(phantom, phantom_cfg['temperature'])
    records = []
    for idx_sphere, (ref, meas) in enumerate(zip(reference, measured), 1):
        if idx_sphere <= phantom_cfg['n_dropped_vials']:
            continue
        records.append({'phantom': phantom, 'study_name': study, 'idx_sphere': idx_sphere, 'ref_T1': ref,
                        'meas_T1': meas, 'abs_err_T1': meas - ref, 'rel_err_T1': (meas - ref) / ref * 100,
                        'session': parameters['sessions'][study_cfg['session']], 'vfa_series': study_cfg['vfa'],
                        'b1_series': study_cfg['b1']})
    logging.info(f'{phantom} {study}: {measured.tolist()}')
    return records


def fit_phantom_t1(parameters):
    output_folder = parameters['output_folder']
    os.makedirs(os.path.join(output_folder, 'masks'), exist_ok=True)
    n_flip_angles = parameters['fit']['n_flip_angles']

    records = []
    for phantom, studies in parameters['studies'].items():
        sessions = {}
        vfa_cache = {}
        jobs = []
        for study, study_cfg in studies.items():
            patient_name = parameters['sessions'][study_cfg['session']]
            if patient_name not in sessions:
                sessions[patient_name] = find_session(parameters['data_root'], patient_name)
            session_index = sessions[patient_name]

            key = (patient_name, study_cfg['vfa'])
            if key not in vfa_cache:
                series = [session_index[study_cfg['vfa'] + 100 * k] for k in range(n_flip_angles)]
                vfa = load_vfa_series(series)
                masks, detection_image = make_vial_masks(vfa[1], parameters['phantoms'][phantom])
                name = f'{phantom}_{patient_name}_vfa{study_cfg["vfa"]}'
                save_mask(masks, vfa[2], os.path.join(output_folder, 'masks', f'{name}_vial_mask.nii.gz'))
                vfa_cache[key] = (vfa, masks, detection_image, name)
            vfa, masks, detection_image, name = vfa_cache[key]
            jobs.append(delayed(fit_study)(phantom, study, study_cfg, vfa, masks, detection_image, name,
                                           session_index, parameters))

        for study_records in Parallel(n_jobs=parameters['fit']['n_jobs'])(jobs):
            records += study_records

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(output_folder, 'fitted_values_T1.csv'), index=False)
    logging.info(f'saved {os.path.join(output_folder, "fitted_values_T1.csv")}')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameterfile', type=str, default='parameters/vfa_phantom_paper.yml')
    parser.add_argument('--data_root', type=str, default=None, help='folder with the DICOM data, overrides the parameterfile')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    with open(args.parameterfile) as file:
        parameters = yaml.load(file, Loader=yaml.Loader)
    if args.data_root is not None:
        parameters['data_root'] = args.data_root
    if not os.path.isdir(parameters['data_root']):
        raise SystemExit(f"Data folder {parameters['data_root']} not found; download the data and pass --data_root")
    fit_phantom_t1(parameters)
