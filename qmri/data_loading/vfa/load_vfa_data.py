import os
import re
import logging
import numpy as np
import torch
import torchio as tio


def load_vfa_subject(study_subject_path):
    """
    Loads a TorchIO subject with VFA data (signals and flip angles).
    Assumes the data is stored in a folder with the following structure:

    study_subject_path
    ├──flip angle AA
       └─ *vfaAA*.nii.gz (3D data)
    ├──flip angle BB
       └─ *vfaBB*.nii.gz (3D data)

    Args:
        study_subject_path: Path where the subject data is located.

    Returns:
        TorchIO subject: A subject containing VFA signals and flip angles.
    """
    subject_dict = {}
    xvals = []
    signals = []

    for flip_angle_dir in sorted(os.listdir(study_subject_path)):
        flip_angle_path = os.path.join(study_subject_path, flip_angle_dir)

        if os.path.isdir(flip_angle_path):
            nii_files = [f for f in os.listdir(flip_angle_path) if f.endswith(".nii.gz")]

            if len(nii_files) != 1:
                raise ValueError(f"Expected exactly one .nii.gz file in {flip_angle_path}, found {len(nii_files)}.")

            nii_file_path = os.path.join(flip_angle_path, nii_files[0])

            # Load image and preprocess
            image = tio.ScalarImage(nii_file_path)
            image.set_data(image.data.to(dtype=torch.float32))

            # Extract flip angle
            match = re.search(r"vfa(\d+)", flip_angle_dir)
            if not match:
                raise ValueError(f"Flip angle not found in directory name: {flip_angle_dir}")

            flip_angle = int(match.group(1))
            xvals.append(flip_angle)
            signals.append(image.data.numpy())

    # Ensure consistent flip angles
    if len(set(xvals)) != len(xvals):
        raise ValueError("Duplicate or inconsistent flip angles detected.")

    # Stack signals
    signals = np.stack(signals, axis=0)
    if signals.ndim == 3:
        signals = np.expand_dims(signals, axis=1)

    # Prepare xvals as a TorchIO Image
    xvals_tensor = torch.tensor(np.array(xvals).reshape(len(xvals), 1, 1, 1), dtype=torch.float32)
    subject_dict["xvals"] = tio.ScalarImage(tensor=xvals_tensor)

    # Prepare signals as a TorchIO Image
    signals_tensor = torch.tensor(signals, dtype=torch.float32)
    subject_dict["signals"] = tio.ScalarImage(tensor=signals_tensor)

    return tio.Subject(subject_dict)

def load_vfa_subject_4d_nii(study_subject_path):
    """
    Loads a TorchIO subject with VFA data (signals and flip angles).
    Assumes the data is stored in a folder with the following structure:

    study_subject_path
    ├── *.nii.gz (4D (3d + flip angles) data)
    ├── *.txt (flip angles)

    Args:
        study_subject_path: Path where the subject data is located.

    Returns:
        TorchIO subject: A subject containing IVIM data.
    """
    subject_dict = {}

    for file in sorted(os.listdir(study_subject_path)):
        file_path = os.path.join(study_subject_path, file)
        logging.info(f"Start loading data from {file_path}")

        # Check for NIfTI image files
        if file.endswith(".nii.gz") or file.endswith(".nii"):
            # Load NIfTI image
            image = tio.ScalarImage(file_path)
            image.set_data(image.data.to(dtype=torch.float32))
            subject_dict["signals"] = image

        # Check if file contains b-values
        elif file.endswith(".txt"):
            text_file = np.genfromtxt(file_path)
            bvals = np.array(text_file)
            subject_dict["xvals"] = tio.Image(tensor=torch.Tensor(np.reshape(bvals, (bvals.shape[0], 1, 1, 1))))

        else:
            logging.info(f"Skipping file {file_path}, unsupported file extension.")

    if "signals" not in subject_dict or "xvals" not in subject_dict:
        raise ValueError("Missing required signals or b-values for IVIM data.")

    return tio.Subject(subject_dict)


def load_vfa_subject_dicom(study_subject_path):
    """
    Loads a TorchIO subject with VFA data (signals and flip angles).
    Assumes the data is stored in a folder with the following structure:

    study_subject_path
    ├──flip angle AA
       └─ *.dcm (2D data)
    ├──flip angle BB
       └─ *.dcm (2D data)

    Args:
        study_subject_path: Path where the subject data is located.

    Returns:
        TorchIO subject: A subject containing VFA signals and flip angles.
    """
    subject_dict = {}
    xvals = []
    signals = []

    for flip_angle_dir in sorted(os.listdir(study_subject_path)):
        flip_angle_path = os.path.join(study_subject_path, flip_angle_dir)

        if os.path.isdir(flip_angle_path):

            # Load image and preprocess
            image = tio.ScalarImage(flip_angle_path)
            image.set_data(image.data.to(dtype=torch.float32))

            # Extract flip angle
            match = re.search(r"ax_(\d+)", flip_angle_dir)
            if not match:
                raise ValueError(f"Flip angle not found in directory name: {flip_angle_dir}")

            flip_angle = int(match.group(1))
            xvals.append(flip_angle)
            signals.append(image.data.numpy())

    # Ensure consistent flip angles
    if len(set(xvals)) != len(xvals):
        raise ValueError("Duplicate or inconsistent flip angles detected.")

    # Stack signals
    signals = np.stack(signals, axis=0)
    signals = np.squeeze(signals)
    if signals.ndim == 3:
        signals = np.expand_dims(signals, axis=1)

    # Prepare xvals as a TorchIO Image
    xvals_tensor = torch.tensor(np.array(xvals).reshape(len(xvals), 1, 1, 1), dtype=torch.float32)
    subject_dict["xvals"] = tio.ScalarImage(tensor=xvals_tensor)

    # Prepare signals as a TorchIO Image
    signals_tensor = torch.tensor(signals, dtype=torch.float32)
    subject_dict["signals"] = tio.ScalarImage(tensor=signals_tensor)

    return tio.Subject(subject_dict)