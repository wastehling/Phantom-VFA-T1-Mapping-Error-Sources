import os
import logging
import torch
import pickle
import numpy as np
import torchio as tio


def load_ivim_subject(study_subject_path):
    """
    Loads a TorchIO subject with IVIM data (signals and b-values).
    Assumes the data is stored in a folder with the following structure:

    study_subject_path
    ├── *.nii.gz (4D (3d + bvalues) data)
    ├── *.bval (b-values)

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
        elif file.endswith(".bval"):
            text_file = np.genfromtxt(file_path)
            bvals = np.array(text_file)
            subject_dict["xvals"] = tio.Image(tensor=torch.Tensor(np.reshape(bvals, (bvals.shape[0], 1, 1, 1))))

        else:
            logging.info(f"Skipping file {file_path}, unsupported file extension.")

    if "signals" not in subject_dict or "xvals" not in subject_dict:
        raise ValueError("Missing required signals or b-values for IVIM data.")

    return tio.Subject(subject_dict)
