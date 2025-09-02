import os
import logging
import torch
import numpy as np
import torchio as tio


def load_dce_subject(study_subject_path, temporal_resolution):
    """
    loads torchio subject with DCE data (signals and timepoints)

    Assumption: if the data is .nii data, the data is stored in a folder with the following structure:
    study_subject_path
    └── *.nii.gz (4d (3d + timepoints) data)

    Args:
        study_subject_path: path in which subject data is located
        temporal_resolution: time between measurement

    Returns:
        TorchIO subject: A subject containing DCE data.

    """
    subject_dict = {}

    # find all files that match study path and subject id
    signals = {}

    for file in sorted(os.listdir(study_subject_path)):
        file_path = os.path.join(study_subject_path, file)
        logging.info(f'start loading data from {file_path}')

        if file.endswith(".nii.gz"):
            # load nifti image
            image = tio.Image(file_path)
            image.set_data(image.data.to(dtype=torch.float32))
            xvals = np.array(range(1, image.shape[0]+1), dtype=float)
            signals = image.numpy()
            subject_dict["signals"] = tio.Image(tensor=torch.Tensor(signals.astype('float64')))
            xvals *= temporal_resolution
            subject_dict["xvals"] = tio.Image(tensor=torch.Tensor(np.reshape(xvals, (xvals.shape[0], 1, 1, 1))))
        else:
            logging.info(f'skipping loading of file {file_path}, no appropriate file extension. ')

    # Create subject
    return tio.Subject(subject_dict)


def load_t1_map(study_subject_path):
    """
    loads torchio subject with T1 map data

    Assumption: if the data is .nii data, the data is stored in a folder with the following structure:
    study_subject_path
    └── *.nii.gz (4d data)
    Args:
        study_subject_path: path in which subject data is located

    Returns:
        torchio subject: subject containing T1 map data
    """
    # find all files that match study path and subject id
    study_subject_path += '/t1_map/'

    for t1_slice_path in sorted(list(os.listdir(study_subject_path))):
        file_path = os.path.join(study_subject_path, t1_slice_path)
        logging.info(f'start loading data from {file_path}')

        if file_path[-2:] == "gz" or file_path[-2:] == "ii":
            image = tio.Image(file_path)
        else:
            logging.info(f'skipping loading of file {file_path}, no appropriate file extension. ')

    return image


def load_aif_image(subject, simulation_parameters):
    """
    loads torchio image with AIF data
    Args:
        subject: subject to add AIF image to
        simulation_parameters: parameters on how to construct the AIF

    Returns:
        torchio image: image containing AIF data

    """
    # vary the Hct value from 0.3 to 0.6
    Hct = simulation_parameters['aif']['Hct']
    aif = simulation_parameters['aif']['aif'].copy()
    aif['ab'] /= (1 - Hct)
    aif_arr = np.array([aif['t0'], aif['tr'], aif['ab'], aif['mb'], aif['ae'], aif['me'], aif['ar'], aif['mm'],
                        aif['mr']])
    x = subject.get('signals').shape[1] * subject.get('signals').shape[2] * subject.get('signals').shape[3]
    aif_arr = np.repeat(aif_arr[:, np.newaxis], x, axis=1)
    aif_arr = np.reshape(aif_arr, (aif_arr.shape[0], subject.get('signals').shape[1],
                                      subject.get('signals').shape[2], subject.get('signals').shape[3]))
    return tio.Image(tensor=torch.Tensor(aif_arr.astype('float64')))