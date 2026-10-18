"""
Loading of Philips VFA and B1+ map DICOM exports, as used for the VFA phantom paper.

The scanner export contains one folder (or zip file) per series, named '<series number>_<description>', e.g.
'00201_2_FFE_FA15_TR66.zip'. Series are addressed by their series number (201 for the example above).
"""
import io
import os
import re
import zipfile

import numpy as np
import pydicom
from scipy.ndimage import map_coordinates


def index_session(session_path):
    """
    Maps series numbers to their folder or zip file within an exported session.
    Args:
        session_path: folder containing the series of one exam (searched recursively)

    Returns:
        dict: series number -> path of folder or zip file
    """
    index = {}
    for dirpath, dirnames, filenames in os.walk(session_path):
        for name in dirnames + filenames:
            match = re.match(r"^(\d{5})_", name)
            if match and (name.endswith('.zip') or os.path.isdir(os.path.join(dirpath, name))):
                index.setdefault(int(match.group(1)), os.path.join(dirpath, name))
    return index


def _read_datasets(series_path):
    """Reads all DICOM files of a series, either from a folder or from a zip file."""
    if series_path.endswith('.zip'):
        with zipfile.ZipFile(series_path) as archive:
            return [pydicom.dcmread(io.BytesIO(archive.read(name))) for name in archive.namelist()
                    if name.lower().endswith('.dcm')]
    return [pydicom.dcmread(os.path.join(dirpath, name))
            for dirpath, _, filenames in os.walk(series_path) for name in filenames if name.lower().endswith('.dcm')]


def read_dicom_volume(series_path, image_type=None, philips_floating_point=True):
    """
    Reads a (multi-slice) DICOM series into a 3D volume.
    Args:
        series_path: folder or zip file of the series
        image_type: only keep images containing this ImageType value (e.g. 'M_B1'), None keeps all
        philips_floating_point: if True, return Philips floating point values (stored value / private scale slope),
            which are comparable between series. If False, return displayed values (RescaleSlope/Intercept), which
            are e.g. the B1+ maps in percent.

    Returns:
        volume: array [slice, row, column], slices sorted along the slice normal
        affine: 4x4 matrix mapping voxel (column, row, slice) to patient (LPS) coordinates in mm
        dataset: first DICOM dataset of the sorted series, for header information
    """
    datasets = _read_datasets(series_path)
    if image_type is not None:
        datasets = [ds for ds in datasets if image_type in ds.ImageType]
    if not datasets:
        raise ValueError(f'No images (image type {image_type}) found in {series_path}')

    orientation = np.array(datasets[0].ImageOrientationPatient, dtype=float)
    row_dir, col_dir = orientation[:3], orientation[3:]
    normal = np.cross(row_dir, col_dir)
    datasets.sort(key=lambda ds: np.dot(normal, np.array(ds.ImagePositionPatient, dtype=float)))

    slices = []
    for ds in datasets:
        stored = ds.pixel_array.astype(np.float64)
        if philips_floating_point:
            slices.append(stored / float(ds[0x2005, 0x100E].value))
        else:
            slices.append(stored * float(ds.RescaleSlope) + float(ds.RescaleIntercept))
    volume = np.stack(slices)

    origin = np.array(datasets[0].ImagePositionPatient, dtype=float)
    pixel_spacing = np.array(datasets[0].PixelSpacing, dtype=float)
    if len(datasets) > 1:
        slice_spacing = np.dot(normal, np.array(datasets[1].ImagePositionPatient, dtype=float) - origin)
    else:
        slice_spacing = float(datasets[0].SliceThickness)
    affine = np.eye(4)
    affine[:3, 0] = row_dir * pixel_spacing[1]
    affine[:3, 1] = col_dir * pixel_spacing[0]
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = origin
    return volume, affine, datasets[0]


def load_vfa_series(series_paths):
    """
    Loads a VFA acquisition stored as one series per flip angle.
    Args:
        series_paths: list of folders or zip files, one per flip angle

    Returns:
        flip_angles: array [n_fa] in degrees, sorted ascending
        signals: array [n_fa, slice, row, column] of Philips floating point values
        affine: voxel to patient (LPS) affine
        TR: repetition time in ms
    """
    flip_angles, signals, affine, TRs = [], [], None, set()
    for path in series_paths:
        volume, affine_fa, ds = read_dicom_volume(path)
        if affine is not None and not np.allclose(affine, affine_fa, atol=1e-3):
            raise ValueError(f'Geometry of {path} differs from the other flip angles')
        affine = affine_fa
        flip_angles.append(float(ds.FlipAngle))
        TRs.add(round(float(ds.RepetitionTime), 3))
        signals.append(volume)
    if len(TRs) != 1:
        raise ValueError(f'Inconsistent repetition times {TRs} in {series_paths}')
    order = np.argsort(flip_angles)
    return np.array(flip_angles)[order], np.stack(signals)[order], affine, TRs.pop()


def load_b1_map(series_path):
    """
    Loads the B1+ map (image type M_B1, in percent) of a Philips AFI, SDAM or DREAM series.
    Returns:
        volume: B1+ scaling factor (1 = nominal) [slice, row, column]
        affine: voxel to patient (LPS) affine
    """
    volume, affine, _ = read_dicom_volume(series_path, image_type='M_B1', philips_floating_point=False)
    return volume / 100.0, affine


def resample_to_slice(volume, affine, target_affine, target_shape, slice_index, order=1):
    """
    Samples a volume on one slice of a target grid using the patient coordinates of both.
    Single-slice (2D) volumes are extended along their slice normal, i.e. they are used for any target slice within
    the acquired slab.
    Args:
        volume: source volume [slice, row, column]
        affine: source voxel to patient affine
        target_affine: target voxel to patient affine
        target_shape: (rows, columns) of the target slice
        slice_index: slice of the target grid
        order: spline interpolation order

    Returns:
        array [rows, columns], zero outside the source volume
    """
    rows, cols = np.meshgrid(np.arange(target_shape[0]), np.arange(target_shape[1]), indexing='ij')
    target_voxels = np.stack([cols.ravel(), rows.ravel(), np.full(cols.size, slice_index), np.ones(cols.size)])
    source_voxels = np.linalg.inv(affine) @ target_affine @ target_voxels
    source_slice = source_voxels[2] if volume.shape[0] > 1 else np.zeros(cols.size)
    sampled = map_coordinates(volume, [source_slice, source_voxels[1], source_voxels[0]], order=order,
                              mode='constant', cval=0)
    return sampled.reshape(target_shape)
