import json
import os
import logging
import torch
import numpy as np
import torchio as tio


def load_T2star_subject(study_subject_path):
    """
    Loads a TorchIO subject with T2* data (signals and echo times).
    Assumes the data is stored in a folder with the following structure:

    study_subject_path
    ├── *.nii.gz (3D data)
    ├── *.json (Echo times)

    Args:
        study_subject_path: Path where the subject data is located.

    Returns:
        TorchIO subject: A subject containing T2* signals and echo times.
    """
    subject_dict = {}
    image_list = []
    TE_list = []

    for file in sorted(os.listdir(study_subject_path)):
        file_path = os.path.join(study_subject_path, file)
        logging.info(f"Start loading data from {file_path}")

        # Check for NIfTI image files
        if file.endswith(".nii.gz"):
            # Load NIfTI image
            image = tio.ScalarImage(file_path)
            image.set_data(image.data.to(dtype=torch.float32))
            image_list.append(image.data)

            # Load corresponding JSON file for echo times
            json_path = file_path[:-6] + 'json'

            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found for {file_path}")

            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                if "EchoTime" not in data:
                    raise KeyError(f"'EchoTime' key not found in {json_path}")

                TE_list.append(torch.tensor(data["EchoTime"], dtype=torch.float32))

        else:
            logging.info(f"Skipping file {file_path}, unsupported file extension.")

    if not image_list or not TE_list:
        raise ValueError("No valid T2* images or echo times found.")

    # Stack image data and prepare TorchIO Image for signals
    signals_tensor = torch.stack(image_list, dim=0)
    if signals_tensor.dim() == 5:
        signals_tensor = torch.squeeze(signals_tensor)
        logging.info("Signals tensor squeezed to 4D")
    subject_dict["signals"] = tio.ScalarImage(tensor=signals_tensor)

    TE = torch.stack(TE_list, dim=0)
    subject_dict["xvals"] = tio.Image(tensor=torch.Tensor(np.reshape(TE, (TE.shape[0], 1, 1, 1))))

    # Create and return subject
    return tio.Subject(subject_dict)
